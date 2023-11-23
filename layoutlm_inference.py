from datasets import load_dataset
from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor, LayoutLMv2TokenizerFast
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
# model = LayoutLMForTokenClassification.from_pretrained("trng1305/layoutlmv2-sroie-test")
# processor = LayoutLMv2Processor.from_pretrained("trng1305/layoutlmv2-sroie-test")


tokenizer = LayoutLMv2TokenizerFast.from_pretrained("./my-tokenizer")
model = LayoutLMForTokenClassification.from_pretrained("./my-model")
processor = LayoutLMv2Processor.from_pretrained("./my-processor")
# processor.save_pretrained("./my-processor")
# # tokenizer.save_pretrained("./my-tokenizer")



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
        print(prediction, box)
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

    image = Image.open(io.BytesIO(path)).convert("RGB")
    encoding = processor(image, return_tensors="pt")
    # print(encoding)
    del encoding["image"]

    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # get labels
    decoded_texts = []


    myBbox = encoding["bbox"][0]
    myTokenIds = encoding["input_ids"][0]
    
    curLabel = 0
    listToken = {1: [], 2: [], 3: [], 4: []}
    keyExtracted = {}

    
    for i in range(len(myBbox)):
      if predictions[i] == 1:
        listToken[1].append(encoding["input_ids"][0][i])
      elif predictions[i] == 2:
        listToken[2].append(encoding["input_ids"][0][i])
      elif predictions[i] == 3:
        listToken[3].append(encoding["input_ids"][0][i])
      elif predictions[i] == 4:
        listToken[4].append(encoding["input_ids"][0][i])



    def handleTotal(listToken): 
      if (len(listToken) == 0):
        return tokenizer.decode(listToken)
      else:
        listText = []
        for token in listToken:
          listText.append(tokenizer.decode(token))
      print(listText)

    keyExtracted[id2label[1]] = tokenizer.decode(listToken[1]).upper()
    keyExtracted[id2label[2]] = tokenizer.decode(listToken[2]).upper()
    keyExtracted[id2label[3]] = tokenizer.decode(listToken[3]).upper()
    keyExtracted[id2label[4]] = tokenizer.decode(listToken[4]).upper()
    handleTotal(listToken[4])
    print(keyExtracted)
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return keyExtracted, draw_boxes(image, encoding["bbox"][0], labels)
    else:
        return labels
    

# run_inference("./000.jpg")