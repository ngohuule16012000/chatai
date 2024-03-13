from flask import Flask, redirect, url_for, request, render_template, jsonify
import pickle
import pandas as pd
import json, datetime
from transformers import PhobertTokenizer, BertForQuestionAnswering
import torch
from collections import Counter

data = pd.read_excel('data\\train_data.xlsx')
file_path = "data\\chat_history.json"

# Đọc dữ liệu từ file JSON
try:
   with open(file_path, "r", encoding='utf-8') as file:
      chat_history = json.load(file)
except FileNotFoundError:
   chat_history = [] 

# Load mô hình
url = 'c:\\Users\\ngohu\\Documents\\DA2\\QAS\\data\\ModelAI\\'
def loadmodel(model):
   with open(model, 'rb') as file:
      model = pickle.load(file)
   return model
classifier, vectorizer = loadmodel(url + 'classifier.pkl'), loadmodel(url + 'vectorizer.pkl')

model_name = "data\\ModelAI\\QASBert"
tokenizerBert = PhobertTokenizer.from_pretrained(model_name)
modelBert = BertForQuestionAnswering.from_pretrained(model_name)

def calculate_f1(true_answer, pred_answer):
   true_answer_tokens = true_answer.lower().split()
   pred_answer_tokens = pred_answer.lower().split()
   common_tokens = Counter(true_answer_tokens) & Counter(pred_answer_tokens)

   # Số lượng tokens chung giữa câu trả lời dự đoán và đúng
   num_same = sum(common_tokens.values())

   if num_same == 0:
      return 0

   precision = 1.0 * num_same / len(pred_answer_tokens)
   recall = 1.0 * num_same / len(true_answer_tokens)
   f1 = (2 * precision * recall) / (precision + recall)

   return f1


def predict_answer(question, contents, model, tokenizer):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   model.eval()

   f1_scores = []
   answers = []

   for content in contents:
      # Tokenize input
      inputs = tokenizer.encode_plus(question, content,  return_tensors="pt", 
         padding=True, 
         truncation= 'only_second',
         max_length=258
      )
      
      # Make prediction
      outputs = model(**inputs)
      start_scores, end_scores = outputs.start_logits, outputs.end_logits

      # Get the most likely answer
      start_index = torch.argmax(start_scores)
      end_index = torch.argmax(end_scores) + 1 # Add 1 because the end index is inclusive

      # Decode the answer from the tokens
      predicted_answer = tokenizer.decode(inputs["input_ids"][0, start_index:end_index], skip_special_tokens=True)
      answers.append(predicted_answer)

      if predicted_answer != "":
         true_answers = data[data['context'] == content]['answers'].tolist() 
         for true_answer in true_answers:
            f1 = calculate_f1(true_answer, predicted_answer)
            if f1 != 0:
               f1_scores.append(f1)
   if len(f1_scores) != 0:
      average_f1_score = sum(f1_scores) / len(f1_scores)
   else:
      average_f1_score = 0.0

   return answers, average_f1_score

app = Flask(__name__)

@app.route('/')

# def hello_name(score):
#    return render_template('hello.html', marks = score)
# app.add_url_rule('/hello/<int:score>', 'hello', hello_name)

# def success(name):
#    return 'welcome %s' % name
# app.add_url_rule('/success/<name>', 'success', success)

# def login():
#    if request.method == 'POST':
#       user = request.form['nm']
#       return redirect(url_for('success',name = user))
#    else:
#       user = request.args.get('nm')
#       return redirect(url_for('success',name = user))
# app.add_url_rule('/login', 'login', login, methods = ['POST', 'GET'])

def chat():
   return render_template('chat.html')
app.add_url_rule('/', 'chat', chat, methods = ['POST', 'GET'])

def chatbox():
   return render_template('chatbox.html')
app.add_url_rule('/chatbox', 'chatbox', chatbox)

def hischatbox():
   # Đọc file JSON
   with open('C:\\Users\\ngohu\\Documents\\DA2\\QAS\\data\\chat_history.json', encoding='utf-8') as json_file:
      messages = json.load(json_file)

   sorted_messages = sorted(messages, key=lambda x: datetime.datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S.%f"), reverse=True)
    
   # Render template và truyền dữ liệu JSON vào template
   return render_template('admin.html', messages=sorted_messages)
app.add_url_rule('/admin', 'admin', hischatbox)

# def send_message():
#    message = request.form["message"]
#    return jsonify({"message": message})
# app.add_url_rule('/send_message', 'send_message', send_message, methods = ['POST'])

def predict_intent(question):
   question_vect = vectorizer.transform([question])  # Chuyển đổi câu hỏi thành vector đặc trưng
   intent = classifier.predict(question_vect)
   return intent

def chatJson(user, bot):
   return {"user_chat": user, "bot_chat": bot, "timestamp": str(datetime.datetime.now())}

def fixString(chatstring):
   # Loại bỏ khoảng trắng, dấu phẩy, dấu chấm ... ở đầu chuỗi
   chatstring = chatstring.lstrip(' .,:?')
   # Viết hoa chữ cái đầu tiên của chuỗi
   chatstring = chatstring.capitalize()
   # Loại bỏ dấu phẩy, chấm phẩy ... ở cuối chuỗi và thêm dấu chấm
   chatstring = chatstring.rstrip(' ,;?:.') + '.'
   return chatstring

def classify_message():
   message = request.form["message"]

   prediction = predict_intent(message)[0]
   contents = data[data['Title'] == prediction]['context'].drop_duplicates().tolist()
   
   answers, f1_scores = predict_answer(message, contents, modelBert, tokenizerBert)

   predict = " ".join([fixString(i) for i in set(filter(None, answers))])
   chat_history.append(chatJson(message, predict))

   with open(file_path, "w", encoding='utf-8') as file:
      json.dump(chat_history, file, ensure_ascii = False, indent=4)
   return jsonify({"prediction": predict, "f1scores": f1_scores})
app.add_url_rule('/classify_message', 'classify_message', classify_message, methods = ['POST'])

if __name__ == '__main__':
   app.run(debug = True)


