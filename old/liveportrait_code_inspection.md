# LivePortrait Code Inspection

Inspection target: official `KlingAIResearch/LivePortrait`, cloned locally to `/tmp/LivePortrait`.

Scope: use LivePortrait only for motion representation ideas. Do not use it as an image-to-video-to-event path. The target remains:

`single RGB image -> semantic mask / blob map -> LivePortrait-inspired facial motion vectors -> direct motion-to-event renderer -> event stream`

## Relevant Files And Functions

### Motion extraction

- `src/modules/motion_extractor.py`
  - `MotionExtractor.forward(x)` calls the ConvNeXtV2 detector and returns the motion dictionary.
- `src/modules/convnextv2.py`
  - The detector heads predict:
    - `kp`: implicit canonical keypoints, `3 * num_kp`
    - `exp`: expression offsets, `3 * num_kp`
    - `pitch`, `yaw`, `roll`: head pose logits
    - `t`: translation
    - `scale`: scalar scale
- `src/live_portrait_wrapper.py`
  - `LivePortraitWrapper.get_kp_info(x)` refines the detector output into:
    - `kp`: `B x N x 3`
    - `exp`: `B x N x 3`
    - `pitch/yaw/roll`: degrees, `B x 1`
    - `t`, `scale`
  - `LivePortraitWrapper.transform_keypoint(kp_info)` computes transformed implicit keypoints:
    - `x = scale * (kp @ R + exp) + t_xy`

The model config sets `num_kp: 21`, so the central sparse motion representation is 21 implicit 3D keypoints plus 21 expression vectors.

### Motion templates

- `src/live_portrait_pipeline.py`
  - `LivePortraitPipeline.make_motion_template(...)` stores per-frame motion:
    - `scale`
    - `R`
    - `exp`
    - `t`
    - `kp`
    - `x_s`
    - eye close ratios
    - lip close ratios

This is the cleanest evidence that LivePortrait's reusable motion signal is not RGB video. Its `.pkl` motion templates contain keypoint/pose/expression state and eye/lip ratios.

### Eye and lip retargeting

- `src/utils/retargeting_utils.py`
  - `calc_eye_close_ratio(lmk)` computes left/right eye closure from 2D landmarks.
  - `calc_lip_close_ratio(lmk)` computes lip closure from 2D landmarks.
- `src/live_portrait_wrapper.py`
  - `calc_combined_eye_ratio(c_d_eyes_i, source_lmk)` builds `[source_left_eye, source_right_eye, target_eye]`, shape `B x 3`.
  - `calc_combined_lip_ratio(c_d_lip_i, source_lmk)` builds `[source_lip, target_lip]`, shape `B x 2`.
  - `retarget_eye(kp_source, eye_close_ratio)` runs the eye retargeting MLP and returns `B x 21 x 3` keypoint deltas.
  - `retarget_lip(kp_source, lip_close_ratio)` runs the lip retargeting MLP and returns `B x 21 x 3` keypoint deltas.
- `src/modules/stitching_retargeting_network.py`
  - `StitchingRetargetingNetwork` is a small MLP used for stitching, eye retargeting, and lip retargeting.
- `src/config/models.yaml`
  - `eye`: input size 66 = `21*3 + 3`, output size 63 = `21*3`.
  - `lip`: input size 65 = `21*3 + 2`, output size 63 = `21*3`.

The retargeting modules operate directly on the transformed source implicit keypoints and ratio controls. Their outputs are sparse motion deltas, not frames.

### Stitching / motion correction

- `src/live_portrait_wrapper.py`
  - `stitch(kp_source, kp_driving)` runs the stitching MLP on concatenated source/driving keypoints.
  - `stitching(kp_source, kp_driving)` splits the MLP output into:
    - `delta_exp`: `B x 21 x 3`
    - `delta_tx_ty`: `B x 1 x 2`
  - It adds these to the driving keypoints to reduce misalignment.
- `src/config/models.yaml`
  - `stitching`: input size 126 = `(21*3)*2`, output size 65 = `21*3 + 2`.

For our purposes, stitching is best treated as a learned sparse motion regularizer over keypoints, not as a paste-back or RGB blending operation.

### Animation-region motion transformation

- `src/live_portrait_pipeline.py`
  - The main animation loop constructs `x_d_i_new`, the target driving keypoints for the current frame.
  - It supports relative motion:
    - expression: `source_exp + (driving_exp_i - driving_exp_0)`
    - pose: `(R_d_i @ R_d_0.T) @ R_s`
    - translation: `source_t + (driving_t_i - driving_t_0)`
    - scale ratio transfer when `animation_region == "all"`
  - It supports regional controls:
    - lip indices: `[6, 12, 14, 17, 19, 20]`
    - eye indices: `[11, 13, 15, 16, 18]`
    - broader expression indices: `[1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]`, plus special handling for selected dimensions of indices `3:5`, `5`, `8`, and `9`.
  - Final keypoint transform:
    - `x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new`
  - Final intensity control:
    - `x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier`

This is directly relevant to a motion-vector renderer: the useful output is `x_d_i_new - x_s` and optionally its regional decomposition.

### Dense warping

- `src/modules/warping_network.py`
  - `WarpingNetwork.forward(feature_3d, kp_driving, kp_source)` calls `DenseMotionNetwork`.
  - It returns:
    - `deformation`: `B x 16 x 64 x 64 x 3`
    - `occlusion_map`: `B x 1 x 64 x 64`
    - `out`: warped feature map before the SPADE decoder
- `src/modules/dense_motion.py`
  - `create_sparse_motions(...)` builds per-keypoint 3D coordinate grids from source and driving keypoints.
  - `create_heatmap_representations(...)` builds driving-minus-source Gaussian heatmaps.
  - `forward(...)` predicts a soft mask over background plus 21 keypoint motions, then blends sparse motions into:
    - `deformation`: `B x D x H x W x 3`, where default `D=16`, `H=64`, `W=64`.
    - `mask`: `B x 22 x 16 x 64 x 64`.
    - optional `occlusion_map`: `B x 1 x 64 x 64`.
- `src/live_portrait_wrapper.py`
  - `warp_decode(...)` obtains the warping output first, then passes only `ret_dct['out']` through the RGB generator.

The dense warping field is accessible before RGB decoding. However, it is a feature-volume sampling grid, not a 2D optical flow field in source image coordinates. It can inspire our blob-level deformation, but should not be treated as ready-to-render event motion without calibration.

## Available Motion Representation

LivePortrait exposes three useful levels:

1. Sparse canonical motion:
   - `kp`: canonical 21 implicit 3D keypoints.
   - `exp`: 21 expression-offset vectors.
   - `R`, `scale`, `t`: pose/global transform.
   - `x_s`: transformed source keypoints.
   - `x_d_i_new`: transformed target keypoints after expression, pose, retargeting, stitching, and multiplier.

2. Retargeting deltas:
   - `eyes_delta`: learned `B x 21 x 3` sparse keypoint offsets from source eye ratios and target eye ratio.
   - `lip_delta`: learned `B x 21 x 3` sparse keypoint offsets from source lip ratio and target lip ratio.
   - `stitching` deltas: learned `B x 21 x 3` offsets plus global `tx/ty`.

3. Dense feature-space deformation:
   - `deformation`: `B x 16 x 64 x 64 x 3`.
   - `mask`: soft assignment over 21 keypoints plus background.
   - `occlusion_map`: 2D visibility-like map.

## Are Dense Warping Fields Accessible?

Yes. `warp_decode(...)` returns `ret_dct` containing `deformation` and `occlusion_map` before parsing RGB output.

Important caveats:

- The deformation is a normalized grid for `torch.nn.functional.grid_sample`, not pixel displacement.
- It is 3D feature-volume motion at `16 x 64 x 64`, while the generated RGB output is produced later by the SPADE decoder.
- The 2D event renderer should not consume it as-is. A safer first integration is to consume sparse keypoint deltas and optionally use the dense field only as a prior for local blob influence weights.

## Mapping LivePortrait Motion To Semantic Blobs

Our semantic mask / blob map can use LivePortrait motion as a sparse control signal:

- Treat each semantic blob as an anchor with:
  - semantic class: eye, upper eyelid, lower eyelid, lip, cheek, jaw, brow, nose, background
  - center and covariance in image coordinates
  - associated LivePortrait keypoint subset
- Project or fit the 21 implicit keypoints into crop/image coordinates using `x_s[..., :2]`.
- For each blob, compute motion as a weighted blend:
  - `blob_motion = sum_j w(blob, kp_j) * ((x_d_i_new[j, :2] - x_s[j, :2]) * scale_to_pixels)`
- Use regional gating:
  - eye blobs primarily use indices `[11, 13, 15, 16, 18]` and `eyes_delta`.
  - lip blobs primarily use `[6, 12, 14, 17, 19, 20]` and `lip_delta`.
  - head / face contour blobs use pose, scale, translation, and broader expression indices.
- Use `stitching` output only as a correction to sparse motion, not as paste-back.
- Optionally derive local uncertainty/visibility:
  - use the dense `mask` or `occlusion_map` as a prior for blob confidence once calibrated.

The most useful event-renderer input is a per-blob 2D velocity field over a time interval, not generated RGB frames.

## Recommended Integration Plan For I2E

1. Build a LivePortrait-motion inspection adapter only.
   - Inputs: one cropped RGB face and optional control values for pose, eye ratio, lip ratio, expression multiplier.
   - Outputs: `kp`, `exp`, `R`, `scale`, `t`, `x_s`, `x_d`, `x_d - x_s`, `eyes_delta`, `lip_delta`, optional `deformation`.
   - Do not call `spade_generator` or save video.

2. Start with sparse motion.
   - Use `get_kp_info`, `transform_keypoint`, `retarget_eye`, `retarget_lip`, and `stitching`.
   - Stop before `warp_decode`, unless collecting deformation for analysis.
   - Convert `x_d - x_s` into semantic blob motion vectors.

3. Add semantic blob binding.
   - Define blob-to-keypoint association tables for eyes, lips, brows, jaw/cheeks.
   - Use the LivePortrait regional index sets as the initial binding.
   - Learn or tune weights from mask geometry rather than relying on implicit keypoint semantics alone.

4. Add direct event rendering.
   - Move blob masks analytically using the per-blob motion vectors.
   - Render event polarity from intensity/edge changes induced by blob motion, not from generated RGB video.
   - Keep the RGB source image as initial appearance only.

5. Use dense deformation only as an optional second-stage prior.
   - Inspect `deformation`, `mask`, and `occlusion_map` for consistency with sparse blob motion.
   - If useful, downsample or aggregate the dense field over semantic masks to refine blob motion.
   - Avoid treating LivePortrait's feature-volume deformation as physical optical flow without validation.

6. Keep LivePortrait isolated.
   - Vendor or wrap only the minimal inference pieces needed for motion extraction.
   - Avoid integrating LivePortrait's RGB generator as part of the event pipeline.
   - Keep the interface motion-centric so the I2E renderer remains direct.

## Bottom Line

LivePortrait is useful for our target method as a learned sparse facial-motion representation:

- 21 implicit 3D keypoints.
- expression offsets.
- pose/scale/translation.
- learned eye/lip retargeting deltas.
- learned stitching correction.
- optional dense feature-space deformation and occlusion.

The strongest integration path is to extract `x_d - x_s` and retargeting deltas, bind them to semantic blobs, and render events directly from blob motion. The dense warping field is accessible but should be treated as an analysis signal or optional prior, not as the primary event-rendering path.
