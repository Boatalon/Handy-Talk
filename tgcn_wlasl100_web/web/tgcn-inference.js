import * as ort from "onnxruntime-web";

function softmax(values) {
  let maximum = -Infinity;
  for (const value of values) maximum = Math.max(maximum, value);

  const probabilities = new Float32Array(values.length);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const probability = Math.exp(values[index] - maximum);
    probabilities[index] = probability;
    total += probability;
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= total;
  }
  return probabilities;
}

function bestPredictions(probabilities, labels, count) {
  return Array.from(probabilities, (probability, index) => ({
    index,
    label: labels[index],
    probability,
  }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, count);
}

export async function createTgcnRecognizer({
  modelUrl = "/models/tgcn_wlasl100.onnx",
  labelsUrl = "/models/labels.json",
  executionProviders = ["wasm"],
} = {}) {
  const labelsResponse = await fetch(labelsUrl);
  if (!labelsResponse.ok) {
    throw new Error(`Could not load labels: ${labelsResponse.status}`);
  }
  const labels = await labelsResponse.json();
  if (!Array.isArray(labels) || labels.length !== 100) {
    throw new Error("Expected exactly 100 WLASL labels.");
  }

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders,
    graphOptimizationLevel: "all",
  });

  return {
    labels,
    session,

    /**
     * @param {Float32Array} keypoints - Flattened tensor with shape [1,55,100].
     * @param {number} topK - Number of ranked predictions to return.
     */
    async predict(keypoints, topK = 5) {
      if (!(keypoints instanceof Float32Array) || keypoints.length !== 5500) {
        throw new Error("keypoints must be Float32Array(5500) for [1,55,100].");
      }
      const input = new ort.Tensor("float32", keypoints, [1, 55, 100]);
      const outputs = await session.run({ keypoints: input });
      const probabilities = softmax(outputs.logits.data);
      return bestPredictions(probabilities, labels, topK);
    },
  };
}
