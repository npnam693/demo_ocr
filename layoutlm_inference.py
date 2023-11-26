      # elif predictions[i] == 4 or (i > 0 and torch.equal(myBbox[i-1],myBbox[i]) and predictions[i-1] == 4):
from datasets import load_dataset
from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor, LayoutLMv2TokenizerFast
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from preprocess import preprocess_image
import cv2
import numpy as np
import torch
# tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
# model = LayoutLMForTokenClassification.from_pretrained("trng1305/layoutlmv2-sroie-test")
# processor = LayoutLMv2Processor.from_pretrained("trng1305/layoutlmv2-sroie-test")


tokenizer = LayoutLMv2TokenizerFast.from_pretrained("./my-tokenizer")
model = LayoutLMForTokenClassification.from_pretrained("./my-model")
processor = LayoutLMv2Processor.from_pretrained("./my-processor")


def unnormalize_box(bbox, width, height):
  return [
      width * (bbox[0] / 1000),
      height * (bbox[1] / 1000),
      width * (bbox[2] / 1000),
      height * (bbox[3] / 1000),
  ]
label2color = {
    "O": "blue",
    "B-COMPANY": "red",
    "I-COMPANY": "green",
    "B-DATE": "blue",
    "I-DATE": "yellow",
    "I-ADDRESS": "pink",
    "B-ADDRESS": "black",
    "I-TOTAL": "purple",
    "B-TOTAL": "grey",
}
# draw results onto the image
def draw_boxes(image, boxes, predictions):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        # print(prediction, box)
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)


    image_buffer = io.BytesIO()
    image.save(image_buffer, format='PNG')  # Convert to PNG (or other format if needed)
    image_buffer.seek(0)

    # Encode the image bytes to base64 string
    base64_image = base64.b64encode(image_buffer.getvalue()).decode()
    return base64_image

def run_inference(path, model=model, processor=processor, output_image=True):
    labels = ['O', 'company', 'date', 'address', 'total']
    id2label = {v: k for v, k in enumerate(labels)}

    # image = Image.open(path)
    image = Image.open(io.BytesIO(path))
    image = preprocess_image(np.array(image))
    encoding = processor(image, return_tensors="pt")
    # print(encoding)
    del encoding["image"]

    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    bBoxs = encoding["bbox"][0]
    listToken = {1: [], 2: [], 3: [], 4: []}
    keyExtracted = {}
    curBbox = None
    curLabel = 0
    # print(curBbox)

    for i in range(len(bBoxs)):
      if curBbox is not None and predictions[i] == 0 and torch.equal(curBbox,bBoxs[i]) and curLabel != 0:
        if curLabel == 4:
           listToken[curLabel][-1].append(encoding["input_ids"][0][i])
        else:
          listToken[curLabel].append(encoding["input_ids"][0][i])
      elif predictions[i] == 0:
        curBbox = None
        curLabel = 0
      else:
        if predictions[i] == 4:
          if curBbox is None or not torch.equal(curBbox, bBoxs[i]):
            listToken[predictions[i]].append([encoding["input_ids"][0][i]])
          else:
             listToken[predictions[i]][-1].append(encoding["input_ids"][0][i])
        else:
          listToken[predictions[i]].append(encoding["input_ids"][0][i])
        curBbox = bBoxs[i]
        curLabel = predictions[i]

    # print(listToken)
    keyExtracted[id2label[1]] = tokenizer.decode(listToken[1]).upper()
    keyExtracted[id2label[2]] = tokenizer.decode(listToken[2]).upper().replace(" ", "")
    keyExtracted[id2label[3]] = tokenizer.decode(listToken[3]).upper()
    keyExtracted[id2label[4]] = [tokenizer.decode(token).upper().replace(" ", "") for token in listToken[4]]
    print(keyExtracted)
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return keyExtracted, draw_boxes(image, encoding["bbox"][0], labels)
    else:
        return labels


# run_inference("./016.jpg")