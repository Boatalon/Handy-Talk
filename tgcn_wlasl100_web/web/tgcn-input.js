// Converts MediaPipe Holistic landmarks to the input layout used by the
// original WLASL100 Pose-TGCN model.

export const TGCN_FRAME_COUNT = 50;
export const TGCN_NODE_COUNT = 55;
export const TGCN_FEATURES_PER_NODE = TGCN_FRAME_COUNT * 2;

const MISSING_POINT = Object.freeze([-1, -1]);

function hasCoordinates(landmark) {
  return Boolean(
    landmark &&
      Number.isFinite(landmark.x) &&
      Number.isFinite(landmark.y),
  );
}

function normalizedPoint(landmark) {
  if (!hasCoordinates(landmark)) {
    return [...MISSING_POINT];
  }

  // WLASL preprocessing resized frames to 256 px and applied:
  //   2 * ((coordinate / 256) - 0.5)
  // MediaPipe already gives coordinates relative to image width/height.
  return [2 * landmark.x - 1, 2 * landmark.y - 1];
}

function midpoint(first, second) {
  if (!hasCoordinates(first) || !hasCoordinates(second)) {
    return [...MISSING_POINT];
  }
  const a = normalizedPoint(first);
  const b = normalizedPoint(second);
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

function mapHand(landmarks) {
  const output = [];
  for (let index = 0; index < 21; index += 1) {
    output.push(normalizedPoint(landmarks?.[index]));
  }
  return output;
}

/**
 * Returns one [55][2] frame in the original OpenPose node order:
 * 13 upper-body points, 21 left-hand points, then 21 right-hand points.
 */
export function holisticResultToTgcnFrame(result) {
  const pose = result?.poseLandmarks?.[0];
  if (!pose || pose.length < 25) return null;

  const body = [
    normalizedPoint(pose[0]), // nose
    midpoint(pose[11], pose[12]), // neck
    normalizedPoint(pose[12]), // right shoulder
    normalizedPoint(pose[14]), // right elbow
    normalizedPoint(pose[16]), // right wrist
    normalizedPoint(pose[11]), // left shoulder
    normalizedPoint(pose[13]), // left elbow
    normalizedPoint(pose[15]), // left wrist
    midpoint(pose[23], pose[24]), // mid hip
    normalizedPoint(pose[5]), // right eye
    normalizedPoint(pose[2]), // left eye
    normalizedPoint(pose[8]), // right ear
    normalizedPoint(pose[7]), // left ear
  ];

  const leftHand = mapHand(result?.leftHandLandmarks?.[0]);
  const rightHand = mapHand(result?.rightHandLandmarks?.[0]);
  return [...body, ...leftHand, ...rightHand];
}

export class TgcnFrameBuffer {
  constructor(maxFrames = TGCN_FRAME_COUNT) {
    if (maxFrames !== TGCN_FRAME_COUNT) {
      throw new Error(`This model requires exactly ${TGCN_FRAME_COUNT} frames.`);
    }
    this.frames = [];
  }

  get length() {
    return this.frames.length;
  }

  get ready() {
    return this.frames.length === TGCN_FRAME_COUNT;
  }

  clear() {
    this.frames.length = 0;
  }

  pushFrame(frame) {
    if (!Array.isArray(frame) || frame.length !== TGCN_NODE_COUNT) {
      throw new Error(`A TGCN frame must contain ${TGCN_NODE_COUNT} points.`);
    }
    this.frames.push(frame);
    if (this.frames.length > TGCN_FRAME_COUNT) this.frames.shift();
    return this.ready;
  }

  pushHolisticResult(result) {
    const frame = holisticResultToTgcnFrame(result);
    if (!frame) return false;
    return this.pushFrame(frame);
  }

  /** Returns Float32 data laid out as [1, 55, 100]. */
  toFloat32Array() {
    if (!this.ready) {
      throw new Error(
        `Need ${TGCN_FRAME_COUNT} frames; the buffer currently has ${this.frames.length}.`,
      );
    }

    const output = new Float32Array(
      TGCN_NODE_COUNT * TGCN_FEATURES_PER_NODE,
    );

    for (let node = 0; node < TGCN_NODE_COUNT; node += 1) {
      const nodeOffset = node * TGCN_FEATURES_PER_NODE;
      for (let frame = 0; frame < TGCN_FRAME_COUNT; frame += 1) {
        const [x, y] = this.frames[frame][node];
        output[nodeOffset + frame * 2] = x;
        output[nodeOffset + frame * 2 + 1] = y;
      }
    }
    return output;
  }
}
