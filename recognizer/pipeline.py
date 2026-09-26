"""Public pixels-only image-to-diagram API.

Reference PD labels, sample identities and dataset sidecars are deliberately not
imported here. A path is used only to decode image pixels with Pillow.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
from .knotfolio_backend import extract, _color_layers
from .geometry import assemble, beautify, check_embedding
from .diagram import pd_code, validate
from .preprocess import prepare, separate_narrow_contacts
from .recognition_runtime import (DEFAULT_SECONDS, RecognitionDeadline, recognition_runtime,
                                  current, checkpoint, remaining, memoized_attempt)


def recognize(image_path, *, options=None):
    with Image.open(image_path) as image:
        image=ImageOps.exif_transpose(image).convert('RGBA')
        background=Image.new('RGBA',image.size,'white')
        background.alpha_composite(image)
        rgb=np.asarray(background.convert('RGB'))
    return recognize_array(rgb, options=options)


def recognize_array(rgb, *, options=None):
    """Recognize with a shared budget; None explicitly allows an offline search."""
    opts = dict(options or {})
    seconds = opts.pop('time_budget', DEFAULT_SECONDS)
    with recognition_runtime(seconds) as runtime:
        try:
            result = _recognize_with_runtime(rgb, options=opts)
        except RecognitionDeadline:
            result = (runtime.failure[1] if runtime.failure else
                      dict(diagram=None, pd_code=None, diagnostics={}, preview_paths=[],
                           unlinked_unknot_components=0))
            result.update(status='needs_review', diagram=None, pd_code=None,
                          unlinked_unknot_components=None,
                          warnings=['Recognition reached its time budget without a confident diagram. '
                                    'Inspect the marked regions, repair unclear crossings or unwanted ink, and try again.'])
            if runtime.boxes:
                result['diagnostics']['twist_boxes'] = runtime.boxes
            result['diagnostics']['budget_exhausted'] = True
        if result.get('diagram') is None:
            # No complete graph means the number of omitted components is
            # unknown, rather than a claim that there are zero of them.
            result['unlinked_unknot_components'] = None
        result.setdefault('diagnostics', {})['runtime'] = {
            'budget_seconds': runtime.seconds, 'seconds': time.monotonic()-runtime.started,
            'last_stage': runtime.stage, 'cache': dict(runtime.counts)}
        result['diagnostics']['elapsed_seconds'] = time.monotonic()-runtime.started
        return result


def _recognize_with_runtime(rgb, *, options=None):
    started=time.monotonic()
    rgb=np.asarray(rgb)
    if rgb.ndim==2:
        rgb=np.repeat(rgb[:,:,None],3,axis=2)
    if rgb.ndim!=3 or rgb.shape[2] not in (3,4) or min(rgb.shape[:2])<3:
        raise ValueError('Expected an RGB/RGBA or grayscale image at least 3 by 3 pixels.')
    if rgb.dtype.kind not in 'buif':
        raise ValueError('Image pixels must be real numeric values.')
    if not np.isfinite(rgb).all():
        raise ValueError('Image pixels must be finite.')
    if rgb.dtype.kind == 'b':
        # Pillow's 1-bit images decode to boolean arrays: True is white.
        # Casting directly to uint8 instead turns the whole canvas almost black.
        rgb = rgb.astype(np.uint8)*255
    elif np.issubdtype(rgb.dtype,np.floating) and rgb.min()>=0 and rgb.max()<=1:
        rgb=rgb*255
    rgb=np.clip(rgb,0,255).astype(np.uint8)
    if rgb.shape[2]==4:
        # Match Pillow's white-background compositing in recognize(path).
        # Float truncation differs by one level for translucent pixels and
        # can make valid box-review hashes appear stale for the array API.
        alpha=rgb[:,:,3:4].astype(np.uint16)
        rgb=((rgb[:,:,:3].astype(np.uint16)*alpha+255*(255-alpha)+127)//255).astype(np.uint8)
    opts = dict(options or {})
    from .board_photo import MODES
    if opts.get('board_photo', 'auto') not in MODES:
        raise ValueError(f'board_photo must be one of {MODES}.')
    long_gap_retry = opts.pop('long_gap_retry', True)
    stroke_hole_retry = opts.pop('stroke_hole_retry', True)
    learned_ink = opts.pop('learned_ink', False)
    drawn_arrow_retry = opts.pop('drawn_arrow_retry', True)
    style_retry = opts.pop('style_retry', True)
    board_stroke_retry = opts.pop('board_stroke_retry', True)
    box_review = opts.pop('twist_boxes', 'auto')
    if box_review == 'auto':
        from .box_detection import detect_twist_boxes
        checkpoint('locating twist boxes')
        box_review = detect_twist_boxes(rgb, max_ocr_seconds=remaining(12.))
    current().boxes = box_review if isinstance(box_review, dict) else None
    source_rgb = rgb
    photo_labels = None
    if opts.get('remove_labels') and not (isinstance(box_review, dict) and box_review.get('boxes')):
        from .board_photo import looks_photographic
        mode = opts.get('board_photo', 'auto')
        if mode in ('board', 'light', 'dark') or (mode == 'auto' and looks_photographic(rgb)[0]):
            # The old photo early return bypassed the explicit batch-cleanup
            # option. Use the same proposals as the editor, on an owned copy.
            # Default recognition still retains every label. Confirmed boxes
            # take their separate route and cannot lose their coefficients.
            from .label_review import detect_labels, delete_labels
            checkpoint('reviewing requested photo labels')
            markings = detect_labels(rgb)
            candidates = markings['candidates']
            rgb = delete_labels(rgb, markings, [c['id'] for c in candidates])
            opts['remove_labels'] = False
            photo_labels = [{key: c[key] for key in ('id', 'bbox', 'ambiguous')}
                            for c in candidates]
    if isinstance(box_review, dict) and box_review.get('boxes'):
        from .box_recognition import recognize_boxes
        result = recognize_boxes(rgb, box_review, options=opts, long_gap_retry=long_gap_retry)
    else:
        result = _recognize_array(rgb, options=opts)
        # Additional interpretation is a last resort. Keep every existing
        # successful reconstruction, and never reinterpret a confirmed box as
        # ordinary ink. Explicit low-level tracing options remain authoritative.
        default_tracing = (not (set(opts) - {'board_photo', 'remove_labels', 'endpoint_hook_retry'})
                           and not opts.get('remove_labels', False))
        if result.get('diagram') is None and board_stroke_retry and default_tracing:
            from .board_recovery import recognize_board_strokes
            recovered = recognize_board_strokes(rgb, mode=opts.get('board_photo', 'auto'))
            if recovered is not None:
                recovered['diagnostics']['before_board_recovery'] = {
                    'status': result['status'], 'warnings': result.get('warnings', [])}
                result = recovered
        if (result.get('diagram') is None and long_gap_retry and default_tracing
                and opts.get('board_photo', 'auto') in ('auto', 'off')):
            from .long_gaps import recognize_long_underpasses
            recovered = recognize_long_underpasses(rgb)
            if recovered is not None:
                recovered['diagnostics']['before_long_underpass'] = {
                    'status': result['status'],
                    'warnings': result.get('warnings', []),
                }
                result = recovered
        if (result.get('diagram') is None and stroke_hole_retry and default_tracing
                and opts.get('board_photo', 'auto') in ('auto', 'off')):
            from .stroke_holes import recognize_stroke_holes
            recovered = recognize_stroke_holes(rgb)
            if recovered is not None:
                recovered['diagnostics']['before_stroke_holes'] = {
                    'status': result['status'],
                    'warnings': result.get('warnings', []),
                }
                result = recovered
        # The small photo classifier is an explicit experiment. A better ink
        # score alone does not justify replacing the established recognizer.
        if (result.get('diagram') is None and learned_ink and default_tracing
                and opts.get('board_photo', 'auto') != 'off'):
            from .learned_ink import recognize_learned_ink
            recovered = recognize_learned_ink(rgb, mode=opts.get('board_photo', 'auto'))
            if recovered.get('diagram') is not None:
                recovered['diagnostics']['before_learned_ink'] = {
                    'status': result['status'],
                    'warnings': result.get('warnings', []),
                }
                result = recovered
            else:
                result['diagnostics']['learned_ink_attempt'] = {
                    'status': recovered['status'],
                    'model': recovered.get('diagnostics', {}).get('learned_ink'),
                    'attempts': recovered.get('diagnostics', {}).get('attempts', []),
                }
        if (result.get('diagram') is None and drawn_arrow_retry and default_tracing
                and opts.get('board_photo', 'auto') in ('auto', 'off')):
            from .arrow_recovery import recognize_drawn_arrows
            recovered = recognize_drawn_arrows(rgb)
            if recovered is not None:
                recovered['diagnostics']['before_drawn_arrows'] = {
                    'status': result['status'], 'warnings': result.get('warnings', [])}
                result = recovered
        if result.get('diagram') is None and style_retry and default_tracing:
            from .style_recovery import recognize_style_recovery
            recovered = recognize_style_recovery(rgb, mode=opts.get('board_photo', 'auto'))
            if recovered is not None:
                recovered['diagnostics']['before_style_recovery'] = {
                    'status': result['status'], 'warnings': result.get('warnings', [])}
                result = recovered
    from .orientation import apply_source_orientations
    apply_source_orientations(result, source_rgb)
    if photo_labels is not None:
        result['diagnostics']['requested_photo_label_removal'] = photo_labels
        if photo_labels:
            result['warnings'].insert(0, 'Potential labels were removed as explicitly requested; '
                                      'check that no small link component was selected.')
            if result.get('diagram') is not None:
                result['status'] = 'needs_review'
                result['diagram'].setdefault('recognition', {})['warnings'] = result['warnings']
    result['diagnostics']['elapsed_seconds'] = time.monotonic()-started
    return result


@memoized_attempt
def _recognize_array(rgb, *, options=None):
    # Recursive topology repairs never infer orientation from altered pixels.
    # The public boundary uses the original source once, after reconstruction.
    started = time.monotonic()
    original_height,original_width=rgb.shape[:2]
    opts=dict(options or {})
    # Recognition keeps every label by default. The editor offers a separate,
    # explicit review/delete stage; legacy batch cleanup requires opting in.
    remove_labels=opts.pop('remove_labels', False)
    narrow_gap_retry=opts.pop('narrow_gap_retry', True)
    bundle_retry=opts.pop('bundle_retry', True)
    bundle_contact_retry=opts.pop('bundle_contact_retry', True)
    endpoint_hook_retry=opts.pop('endpoint_hook_retry', False)
    drawing_contact_retry=opts.pop('drawing_contact_retry', narrow_gap_retry)
    board_mode=opts.pop('board_photo', 'auto')
    from .board_photo import MODES, looks_photographic, recognize_board
    if board_mode not in MODES:
        raise ValueError(f'board_photo must be one of {MODES}.')
    photographic,detection=looks_photographic(rgb) if board_mode=='auto' and not opts else (False,{})
    if photographic or board_mode in ('board','light','dark'):
        result=recognize_board(rgb,board_mode,opts)
        result['diagnostics']['board_photo']['routing']=detection
        result['diagnostics']['elapsed_seconds']=time.monotonic()-started
        return result
    rgb,preprocessing=prepare(rgb)
    attempts=[]
    # Alternate thresholds act only on image evidence. They cannot consult
    # expected knot type, crossing count, reference PD or other dataset labels.
    configurations=[(rgb,opts,1.)]
    if not opts:
        color_count=_color_layers(rgb.astype(float),rgb.min(axis=2)<205)[1]
        if color_count>1:
            configurations=[(rgb,dict(threshold=t,spur_widths=5.),1.) for t in (100.,150.)]+configurations
            configurations.insert(2,(rgb,dict(threshold=100.,spur_widths=5.,tangent_widths=.7,color_weight=20.),1.))
            if max(rgb.shape[:2])<1000:
                doubled=np.asarray(Image.fromarray(rgb).resize((rgb.shape[1]*2,rgb.shape[0]*2),Image.Resampling.LANCZOS))
                configurations[3:3]=[(doubled,dict(threshold=t,spur_widths=5.,tangent_widths=.7,color_weight=20.,min_object_pixels=1),2.) for t in (100.,60.)]
            configurations.insert(-1,(rgb,dict(threshold=100.,spur_widths=5.,color_weight=20.,min_object_pixels=1,opening_widths=.28),1.))
        if max(rgb.shape[:2])<750 and color_count<=1:
            scale=min(3.,1200/max(rgb.shape[:2]))
            large=np.asarray(Image.fromarray(rgb).resize((round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)),Image.Resampling.LANCZOS))
            configurations.insert(0,(large,dict(threshold=100.,spur_widths=.5,min_object_pixels=1),scale))
        configurations += [(rgb,dict(threshold=t),1.) for t in (150.,100.,230.)]
        configurations.append((rgb,dict(spur_widths=5.),1.))
        if color_count>1:
            # A strand may change color, including around a right-angle bend.
            # Trace the union only after color-aware reconstruction has failed.
            # Existing branch and planarity checks still apply to that union.
            configurations.append((rgb,dict(separate_colors=False),1.))
        if bundle_retry:
            configurations.extend((rgb,dict(bundle_gaps=True,threshold=t,tangent_widths=.7,
                                  stop_at_branches=True,max_endpoints=180),1.)
                                  for t in (100.,150.,205.))
        configurations.extend((rgb,dict(threshold=t,micro_seams=True,arrow_wings=True,
                              stop_at_branches=True,max_endpoints=256),1.)
                              for t in (150.,100.,205.))
    best=None
    bundle_seen=False
    endpoint_floor=None
    contact_traces=[]
    drawing_contacts=[]
    for pixels,config,scale in configurations:
        try:
            # A threshold cutting through antialiasing can turn a few dozen
            # ends into thousands of fragments. Skip matching that candidate;
            # retain the less fragmented masks and continue the fallback.
            if not opts and 'max_endpoints' not in config:
                config={**config,'max_endpoints':max(256,min(512,4*endpoint_floor))
                        if endpoint_floor is not None else 512}
            result=extract(pixels,config)
            count=result['diagnostics']['endpoint_count']
            if count:
                endpoint_floor=min(count,endpoint_floor) if endpoint_floor is not None else count
            bundle_seen |= any(len(m.get('crossings',[]))>=2 for m in result['matches'])
            diagnostic=result['diagnostics']
            if (drawing_contact_retry and not opts and scale==1
                    and 0 < len(diagnostic['branch_points']) <= 4):
                key=tuple(tuple(p) for p in diagnostic['branch_points'])
                if not any(k==key for k,_ in drawing_contacts):
                    drawing_contacts.append((key,result))
            if (bundle_retry and bundle_contact_retry and scale==1 and diagnostic['branch_points']
                    and not diagnostic.get('matching_skipped') and len(diagnostic['branch_points'])<=12):
                key=tuple(tuple(p) for p in diagnostic['branch_points'])
                if not any(k==key for k,_ in contact_traces):
                    contact_traces.append((key,result))
            attempt={'options':config,'scale':scale,'complete':result['complete'],
                     'branches':len(diagnostic['branch_points']),
                     'unmatched':len(diagnostic['unmatched_endpoints']),
                     'conflicts':len(diagnostic['gap_conflicts'])}
            attempts.append(attempt)
            if not result['complete']:
                state = current()
                if state is not None:
                    state.record_trace_failure(result, scale=scale, preprocessing=preprocessing)
                if best is None or sum(attempt[k] for k in ('branches','unmatched','conflicts'))<best[0]:
                    best=(sum(attempt[k] for k in ('branches','unmatched','conflicts')),result,scale)
                continue
            if config.get('bundle_gaps') or endpoint_hook_retry:
                from .endpoint_hooks import normalize_endpoint_hooks
                result,_=normalize_endpoint_hooks(result)
                diagnostic=result['diagnostics']
            diagram=assemble(result,int(pixels.shape[1]),int(pixels.shape[0]))
            if scale!=1:
                for crossing in diagram['crossings']:
                    crossing['point']=[v/scale for v in crossing['point']]
                for edge in diagram['edges']:
                    edge['points']=[[v/scale for v in p] for p in edge['points']]
                diagnostic['stroke_width']/=scale
                for region in diagnostic.get('bundled_crossings',{}).get('regions',[]):
                    for key in ('ends','crossings'):
                        region[key]=[[v/scale for v in p] for p in region[key]]
                if 'bundled_crossings' in diagnostic:
                    diagnostic['bundled_crossings']['review_points']=[[v/scale for v in p]
                        for p in diagnostic['bundled_crossings'].get('review_points',[])]
                for repair in diagnostic.get('endpoint_hook_repairs',[]):
                    for key in ('original_endpoint','joined_endpoint'):
                        repair[key]=[v/scale for v in repair[key]]
            diagram.update(width=original_width,height=original_height)
            diagram=beautify(diagram)
            verdict=validate(diagram)
            if config.get('micro_seams') or config.get('arrow_wings'):
                embedding=check_embedding(diagram,check_strokes=False)
                if not embedding['valid']:
                    raise ValueError('Repaired drawing has an invalid embedding: '+'; '.join(embedding['errors']))
            diagnostic.update(validation=verdict,attempts=attempts,preprocessing=preprocessing,elapsed_seconds=time.monotonic()-started)
            warnings=[]
            if diagnostic.get('endpoint_hook_repairs'):
                warnings.append('Tiny ink-tip hooks were trimmed at inferred joins; inspect the marked endpoints.')
            if diagnostic.get('micro_seam_repairs'):
                warnings.append('Sub-width offsets between opposing pen tips were joined; inspect these short seams.')
            if diagnostic.get('arrow_pruning'):
                warnings.append('Paired arrow wings were separated from the strand trace; their source directions are retained for orientation review.')
            if config.get('bundle_gaps'):
                warnings.append('Parallel-strand gaps were reconstructed using local bundle geometry; inspect the inferred underpasses.')
                if verdict['crossing_free_components']:
                    warnings.append('Isolated closed marks were preserved; check whether these are link components or printed labels.')
            zero_crossing_gaps=sum(not m.get('crossings') for m in result['matches'])
            if zero_crossing_gaps:
                warnings.append(f'{zero_crossing_gaps} small breaks were joined without crossing another strand; inspect these repairs.')
            diagram['recognition']={'stroke_width':diagnostic['stroke_width'],'warnings':warnings}
            return dict(status='needs_review' if warnings else 'ok',diagram=diagram,
                        pd_code=pd_code(diagram),warnings=warnings,
                        unlinked_unknot_components=verdict['crossing_free_components'],diagnostics=diagnostic)
        except (ValueError,RuntimeError,KeyError,IndexError) as exc:
            if attempts:attempts[-1]['error']=str(exc)
            else:attempts.append({'options':config,'error':str(exc)})
    if bundle_retry and bundle_contact_retry and bundle_seen:
        from .bundle_contacts import repair_bundle_contacts
        for _,trace in sorted(contact_traces,key=lambda item:len(item[0]))[:2]:
            cleaned,repair=repair_bundle_contacts(rgb,trace)
            if cleaned is None:
                continue
            repaired=_recognize_array(cleaned,options={**opts,'remove_labels':remove_labels,
                'drawing_contact_retry':drawing_contact_retry,
                'narrow_gap_retry':False,'bundle_contact_retry':False,'bundle_retry':bundle_retry,'board_photo':board_mode})
            if repaired['diagram'] is not None:
                repaired['status']='needs_review'
                repaired['warnings'].insert(0,'An under-strand touching a parallel bundle was separated; inspect the marked crossing region.')
                repaired['diagram']['recognition']['warnings']=repaired['warnings']
                repaired['diagnostics']['preprocessing'].update(repair)
                repaired['diagnostics']['before_bundle_contact_attempts']=attempts
                repaired['diagnostics']['elapsed_seconds']=time.monotonic()-started
                return repaired
    if narrow_gap_retry and best and best[1]['diagnostics'].get('color_layers')==1:
        failed=best[1]['diagnostics']
        scale=best[2]
        points=[[v/scale for v in p] for p in failed.get('branch_points',[])]
        cleaned,repair=separate_narrow_contacts(rgb,points,failed['stroke_width']/scale)
        if cleaned is not None:
            repaired=_recognize_array(cleaned,options={**opts,'remove_labels':False,'narrow_gap_retry':False,
                                     'drawing_contact_retry':drawing_contact_retry,
                                     'bundle_retry':bundle_retry,'bundle_contact_retry':False,
                                     'endpoint_hook_retry':endpoint_hook_retry,'board_photo':board_mode})
            if repaired['diagram'] is not None:
                repaired['status']='needs_review'
                repaired['warnings'].insert(0,'Near-touching strands were separated for recognition; inspect the marked crossings.')
                repaired['diagram']['recognition']['warnings']=repaired['warnings']
                repaired['diagnostics']['preprocessing']={**preprocessing,'narrow_gap_repair':repair}
                repaired['diagnostics']['before_narrow_gap_attempts']=attempts
                repaired['diagnostics']['elapsed_seconds']=time.monotonic()-started
                return repaired
    if drawing_contact_retry and drawing_contacts:
        from .bundle_contacts import repair_bundle_contacts
        for _, trace in sorted(drawing_contacts, key=lambda item:len(item[0]))[:3]:
            cleaned, repair = repair_bundle_contacts(rgb, trace, allow_single=True)
            if cleaned is None:
                continue
            repaired = _recognize_array(cleaned, options={**opts,
                'remove_labels': False, 'drawing_contact_retry': False,
                'narrow_gap_retry': False, 'bundle_contact_retry': False,
                'bundle_retry': bundle_retry, 'board_photo': board_mode})
            if repaired['diagram'] is None or not check_embedding(repaired['diagram'],check_strokes=False)['valid']:
                continue
            repaired['status'] = 'needs_review'
            repaired['warnings'].insert(0, 'A touching under-tip was separated from an uninterrupted strand; inspect the marked crossing.')
            repaired['diagram']['recognition']['warnings'] = repaired['warnings']
            repaired['diagnostics']['preprocessing'].update(repair)
            repaired['diagnostics']['before_drawing_contact_attempts'] = attempts
            repaired['diagnostics']['elapsed_seconds'] = time.monotonic()-started
            return repaired
    if remove_labels:
        from .text_labels import remove_detached_numeric_labels
        cleaned,label_diagnostic=remove_detached_numeric_labels(rgb)
        if not label_diagnostic['removed_labels'] and bundle_retry and bundle_seen:
            from .diagram_labels import remove_detached_labels
            cleaned,label_diagnostic=remove_detached_labels(rgb,max_ocr_calls=12)
        if label_diagnostic['removed_labels']:
            repaired=_recognize_array(cleaned,options={**opts,'remove_labels':False,'narrow_gap_retry':narrow_gap_retry,
                                     'drawing_contact_retry':drawing_contact_retry,
                                     'bundle_retry':bundle_retry,'bundle_contact_retry':bundle_contact_retry,
                                     'endpoint_hook_retry':endpoint_hook_retry or (bundle_retry and bundle_seen),'board_photo':board_mode})
            repaired['diagnostics']['before_label_removal_attempts']=attempts
            repaired['diagnostics']['preprocessing'].update(label_diagnostic)
            repaired['diagnostics']['elapsed_seconds']=time.monotonic()-started
            message='Detached labels were removed automatically; inspect the marked source regions.'
            repaired['warnings'].insert(0,message)
            if repaired['diagram'] is not None:
                repaired['status']='needs_review'
                repaired['diagram']['recognition']['warnings']=repaired['warnings']
            return repaired
    diag=best[1]['diagnostics'] if best else {}
    preview=best[1]['paths'] if best else []
    if best:
        scale=best[2]
        diag['gap_proposals']=[{
            'ends':[[v/scale for v in best[1]['paths'][end[0]]['points'][0 if end[1]==0 else -1]]
                    for end in (m['a'],m['b'])],
            'crossings':[[v/scale for v in h['point']] for h in m['crossings']]}
            for m in best[1]['matches'] if len(m['crossings'])>1]
    if best and best[2]!=1:
        scale=best[2]
        diag['branch_points']=[[v/scale for v in p] for p in diag.get('branch_points',[])]
        for endpoint in diag.get('unmatched_endpoints',[]):
            endpoint['point']=[v/scale for v in endpoint['point']]
        for path in preview:
            path['points']=[[v/scale for v in p] for p in path['points']]
        diag['stroke_width']/=scale
    diag.update(attempts=attempts,preprocessing=preprocessing,elapsed_seconds=time.monotonic()-started)
    warnings=['The image could not be reconstructed as an unambiguous closed classical link. Inspect strand contacts, gaps or extra marks.']
    return dict(status='failed',diagram=None,pd_code=None,warnings=warnings,diagnostics=diag,
                preview_paths=preview,unlinked_unknot_components=0)
