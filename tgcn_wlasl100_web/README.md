# WLASL100 Pose-TGCN for Web

ชุดนี้มี weight ของโมเดล Pose-TGCN สำหรับจำแนกคำ ASL แบบ isolated word จำนวน 100 คำ พร้อม ONNX และตัวแปลงผลลัพธ์จาก MediaPipe Holistic สำหรับใช้ในเว็บ

## ใช้โมเดลนี้ทำอะไรได้

- รับท่าหนึ่งคำต่อหนึ่งช่วงวิดีโอ ไม่ใช่การแปลประโยคต่อเนื่อง
- ใช้ keypoint ช่วงบน 13 จุด + มือซ้าย 21 จุด + มือขวา 21 จุด
- ใช้ 50 เฟรมต่อการทำนาย ควรเก็บใกล้เคียง 25 FPS หรือประมาณ 2 วินาที
- Input ONNX: `keypoints`, shape `[batch, 55, 100]`, `float32`
- Output ONNX: `logits`, shape `[batch, 100]`, `float32`

ไฟล์ `model/labels.json` เรียงตรงกับ index ของ output ตั้งแต่ 0 ถึง 99

## ติดตั้งฝั่งเว็บ

```bash
npm install onnxruntime-web @mediapipe/tasks-vision
```

คัดลอกไฟล์ต่อไปนี้เข้า public assets ของเว็บ:

```text
model/tgcn_wlasl100.onnx  -> public/models/tgcn_wlasl100.onnx
model/labels.json         -> public/models/labels.json
```

ตัวอย่างส่วนที่ต่อหลัง `HolisticLandmarker.detectForVideo(...)`:

```js
import { TgcnFrameBuffer } from "./tgcn-input.js";
import { createTgcnRecognizer } from "./tgcn-inference.js";

const buffer = new TgcnFrameBuffer();
const recognizer = await createTgcnRecognizer({
  modelUrl: "/models/tgcn_wlasl100.onnx",
  labelsUrl: "/models/labels.json",
});

// เรียกส่วนนี้ประมาณ 25 ครั้ง/วินาที
const holisticResult = holisticLandmarker.detectForVideo(video, performance.now());
buffer.pushHolisticResult(holisticResult);

if (buffer.ready) {
  const top5 = await recognizer.predict(buffer.toFloat32Array(), 5);
  console.table(top5);
}
```

ควรแสดงกล้องแบบ mirror ด้วย CSS เท่านั้น อย่ากลับภาพที่ส่งเข้า MediaPipe เพราะทิศแกน x และลำดับมือควรคงเหมือนข้อมูลที่ใช้เทรน

## Mapping ที่ใช้

โมเดลเดิมเทรนด้วย OpenPose ไม่ใช่ MediaPipe โดยตรง ตัวช่วย `web/tgcn-input.js` จึงแปลง MediaPipe Holistic เป็นลำดับเดิมดังนี้:

```text
nose, neck, R shoulder, R elbow, R wrist,
L shoulder, L elbow, L wrist, mid hip,
R eye, L eye, R ear, L ear,
21 left-hand points, 21 right-hand points
```

พิกัด MediaPipe `[0,1]` ถูกแปลงเป็น `[-1,1]` ตาม preprocessing ของ WLASL และจุดที่ตรวจไม่พบใช้ `[-1,-1]` ตามพฤติกรรมของ OpenPose เดิม

## ความแม่นยำและข้อจำกัด

ผล baseline ที่รายงานในงาน WLASL สำหรับ Pose-TGCN รุ่น 100 คำคือ Top-1 55.43%, Top-5 78.68% และ Top-10 87.60% บน test split ของชุดข้อมูล ตัวเลขมากกว่า 85% บนหน้า Hugging Face เป็น self-reported แบบไม่แยกรุ่น จึงไม่ควรใช้เป็นตัวเลขอ้างอิงของรุ่นนี้

การเปลี่ยนจาก OpenPose เป็น MediaPipe ทำให้เกิด domain shift และความแม่นยำจริงจาก webcam อาจต่ำกว่าผลในงานวิจัย ควรเก็บ validation video จากกล้องที่จะใช้จริง และ fine-tune ด้วย MediaPipe keypoint หากต้องการความแม่นยำสำหรับใช้งานจริง

## ไฟล์ในชุด

```text
model/tgcn_wlasl100.onnx                 ONNX สำหรับ browser/backend
model/labels.json                        label array สำหรับ JavaScript
model/label_map.json                     label object แบบ index -> word
pytorch/tgcn_wlasl100_pytorch_model.bin  checkpoint ต้นฉบับ
pytorch/tgcn_model.py                    architecture ต้นฉบับ
pytorch/config.ini                       training/model config
web/tgcn-input.js                        MediaPipe Holistic -> TGCN tensor
web/tgcn-inference.js                    onnxruntime-web inference helper
tools/export_onnx.py                     สคริปต์ export ซ้ำ
```

## ที่มา

- WLASL repository: https://github.com/dxli94/WLASL
- Weight mirror: https://huggingface.co/sharonn18/tgcn-wlasl
- Paper: https://arxiv.org/abs/1910.11006
- MediaPipe Holistic Web guide: https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/web_js

อ่านข้อกำหนดใน `LICENSES.md` ก่อนนำไปเผยแพร่
