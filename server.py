import os,numpy as np
import json
import tempfile
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import librosa,resampy
import pypdf  
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
import speech_recognition as sr
from pydub import AudioSegment  
import cv2
from pymongo import MongoClient
import requests
from flask import Flask, request, jsonify


load_dotenv()  
client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
mongo=os.getenv("MONGO_URL")
youtube_api=os.getenv("YOUTUBE_API_KEY")


app = Flask(__name__)
CORS(app)  
MONGO_URI = mongo

mongo_client = MongoClient(MONGO_URI)
db = mongo_client.videobase
youtube_collection = db.youtube_data

YOUTUBE_API =youtube_api

le = LabelEncoder()
MODEL_PATH = "/models/confidence_voice_model.h5" 
MODEL_PATH1="/models/final_model.h5"
model = load_model(MODEL_PATH)
emotion_model=load_model(MODEL_PATH1)
LABEL_NAMES = ['confident', 'neutral', 'nervous', 'uncertain']

UPLOAD_FOLDER = "uploads"
UPLOAD_AUDIO_FOLDER = "uploads_audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_AUDIO_FOLDER, exist_ok=True)



def get_gemini_response(input_prompt):
    """
    Calls the Gemini API with the given prompt and returns the generated content.
    """

    response = client.models.generate_content(model="gemini-2.5-flash",contents=input_prompt)

    print("Gemini API response:", response)

    if not response.candidates:
        raise ValueError("Gemini API returned no candidates.")

    try:
        candidate_content = response.candidates[0].content.parts[0].text.strip()
        
        json_match = re.search(r'```json\n([\s\S]+?)\n```', candidate_content)
        if json_match:
            json_string = json_match.group(1).strip()
        else:
            json_string = candidate_content

        return json.loads(json_string)
    
    except Exception as e:
        raise ValueError(f"Error extracting or parsing response: {e}")
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  

def process_video_emotions(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % 10 != 0:
            continue

        try:
            processed = preprocess_frame(frame)
            pred = model.predict(processed) 
            emotion_idx = int(np.argmax(pred))

            if emotion_idx == 3:
                status = "happy"
            elif emotion_idx == 5:
                status = "sad"
            elif emotion_idx == 6:
                status = "surprise"
            elif emotion_idx == 1:
                status = "disgust"
            elif emotion_idx == 0:
                status = "angry"
            else:
                status = "neutral"

            predictions.append(status)
        except Exception as e:
            print(f"Frame processing failed: {e}")
            continue

    cap.release()

    if not predictions:
        return "neutral", {}, []

    counts = {emo: predictions.count(emo) for emo in set(predictions)}
    dominant = max(counts, key=counts.get)
    return dominant, counts, predictions

def input_pdf_text(uploaded_file):
    """
    Extracts text from a PDF file.
    """
    reader = pypdf.PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

@app.route('/')
def home():
    return "Hello from PREP-IQ server!"

@app.route('/api/optimize-resume', methods=['POST'])
def optimize_resume():
    """
    Analyzes a resume against a job description and returns a match score and improvement suggestions.
    """
    jd = request.form.get('jd')
    file = request.files.get('resume')

    if not jd or not file:
        return jsonify({"error": "Missing resume or job description."}), 400
    
    try:
        resume_text = input_pdf_text(file)
    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {e}"}), 400

    input_prompt = f"""
    Act as an experienced ATS (Application Tracking System) specialized in evaluating resumes for software engineering, data science, and related fields.
    Assess the resume against the job description and provide a structured JSON response with:
    - JD Match percentage
    - Missing important keywords
    - A short profile summary for improvement.
    
    Resume: {resume_text}
    Job Description: {jd}

    Response Format:
    {{
        "JD Match": "%",
        "MissingKeywords": [],
        "Profile Summary": ""
    }}
    """

    try:
        result = get_gemini_response(input_prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {e}"}), 500


@app.route('/api/evaluate_answer', methods=['POST'])
def evaluate_answer():
    """
    Evaluate a single question/answer pair and return a small JSON with a feedback of how the question could be answered.
    Request JSON: { question: str, answer: str, confidence_label: str, job_position: str (optional) }
    Response JSON: { feedback: str, score: 0-10 (number)}
    """
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")
    confidence_label = data.get("confidence_label", "")

    if not question or not answer:
        return jsonify({"error": "Missing question or answer."}), 400


    prompt = f"""
    You are an interview coach. Evaluate the candidate answer to the following question.Dont be so strict.
    Note: The answer was transcribed using the speechRecognition library, so it may contain minor errors or inconsistencies.
    Question: {question}

    Answer: {answer}

    Provide a concise evaluation using the exact output format below:

    Output Format json containing the following fields:
    ---

    Feedback: [Briefly mention strengths and weaknesses]
    Suggestions: [Actionable advice to improve the answer]
    Score: [Rate from 1 to 10 based on clarity, accuracy]
    ---

    Keep your response short, clear, and to the point. Do not add any extra commentary outside the specified format.
"""


    try:
        resp = get_gemini_response(prompt)
        # Expect resp to be JSON-like; if it's string parse/convert as needed.
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": f"Gemini error: {e}"}), 500


@app.route('/api/get_overall_result', methods=['POST'])
def get_overall_result():
    """
    Compose an overall result, based on:
      - questions[], answers[]
      - confidence_labels[]
      - evaluations[] (per-question small feedback + numeric score)
      - aggregates { avgAnswerScore, avgConfidencePct, resultScore } (optional)
    Return a structured overall feedback JSON.
    """
    data = request.get_json()
    confidence_labels = data.get("confidence_labels", [])
    emotion_labels=data.get("emotions",[])
    evaluations = data.get("evaluations", [])
    scores=data.get("scores",[]) 

    prompt = f"""
    You are a professional interview coach. Given the following evaluated feedbacks for interview questions, produce an overall evaluation of the candidate's performance.
    evaluated feedbacks:{evaluations}
    technical_scores:{scores}
    confidence_labels:{confidence_labels}
    emotion:{emotion_labels}
    Instructions:

    1. Compute a "Confidence Score" out of 10 based on the detected confidence labels,emotions from voice and emotion in the per-question feedbacks.
    2. Compute a "Technical Score" out of 10 based on the quality of the answers in the per-question feedbacks.
    3. Compute an overall score as a weighted average giving **maximum weight to Technical Score**.
    4. Provide an "Areas to Improve" list (3-6 concise points) covering communication, confidence, and technical knowledge.
    5. Provide an "Overall Feedback" paragraph summarizing the candidate's performance.
    6. Provide "tags" which is python list of the format ["","",""].Each tag should represent the area of improvement needed like OOPS,DBMS,Frontend.If confidence is lacking provide me Confidence as a tag too.

   

    Return **JSON only** in this structure:

    {{
    "areas_to_improve": ["...","..."],
    "overall_feedback": "string",
    "confidence_score": number,
    "technical_score": number,
    "overall_score": number,
    "tags":["...","..."]
    }}
    """

    try:
        overall = get_gemini_response(prompt)
        if not overall or not isinstance(overall, dict):

            overall = {
                "areas_to_improve": [],
                "overall_feedback": "Unable to generate feedback.",
                "confidence_score": 0,
                "technical_score": 0,
                "overall_score": 0,
                "tags": []
            }
        return jsonify(overall)
    except Exception as e:
        return jsonify({
        "areas_to_improve": [],
        "overall_feedback": f"Error generating feedback: {e}",
        "confidence_score": 0,
        "technical_score": 0,
        "overall_score": 0,
        "tags": []
    }), 500



@app.route("/api/upload_resume", methods=["POST"])
def upload_resume():
    """
    Uploads a resume and generates 2 interview questions based on the job position.
    """
    if "file" not in request.files or "job_position" not in request.form:
        return jsonify({"error": "Missing file or job position"}), 400

    pdf_file = request.files["file"]
    job_position = request.form["job_position"]
    file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(file_path)
    
    try:
        extracted_text = input_pdf_text(file_path)
    except Exception as e:
        return jsonify({"error": f"Error extracting text: {e}"}), 500

    prompt = f"""
    Act as a seasoned technical interviewer.
    Analyze the given resume in the form of text and the job position, then generate 5 targeted interview questions.
    
    Resume:
    
    {extracted_text}

    Output Format:
    ["question1", "question2", "question3", "question4", "question5"]

    Job Position: {job_position}
    """

    try:
        questions_array = get_gemini_response(prompt)
        return jsonify({"interview_questions": questions_array})
    except Exception as e:
        print(e)
        return jsonify({"error": f"Error generating questions: {e}"}), 500
    
def extract_mfcc(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

def predict_confidence_dl(model, audio_path):
    """
    Runs your CNN on audio_path and returns:
      - label: one of LABEL_NAMES
      - score: the softmax probability of that label
    """

    mfcc = extract_mfcc(audio_path)            
    mfcc = mfcc[np.newaxis, ..., np.newaxis]    


    probs = model.predict(mfcc)[0]    


    idx   = int(np.argmax(probs))
    label = LABEL_NAMES[idx]                    
                 

    return label


@app.route("/api/process_media", methods=["POST"])
def process_media():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    os.makedirs("uploads", exist_ok=True)
    temp_dir = tempfile.mkdtemp()  # creates a safe temporary directory
    audio_webm = os.path.join(temp_dir, "temp_audio.webm")
    audio_wav = os.path.join(temp_dir, "temp_audio.wav")
    request.files["audio"].save(audio_webm)

    try:
        audio = AudioSegment.from_file(audio_webm, format="webm")
        audio.export(audio_wav, format="wav")
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {e}"}), 500

    recognizer = sr.Recognizer()
    transcription = ""
    try:
        with sr.AudioFile(audio_wav) as src:
            recognizer.adjust_for_ambient_noise(src)
            data = recognizer.record(src)
            transcription = recognizer.recognize_google(data)
    except sr.UnknownValueError:
        transcription = ""
    except Exception as e:
        return jsonify({"error": f"Speech API error: {e}"}), 500

    try:
        audio_label = predict_confidence_dl(model, audio_wav)  # your audio model
    except Exception as e:
        audio_label = "neutral"

    emotion_label = "neutral"
    if "video" in request.files:
        video_path = os.path.join("temp_dir", "temp_video.webm")
        request.files["video"].save(video_path)
        try:
            dominant_emotion, _, _ = process_video_emotions(video_path,emotion_model)

            if dominant_emotion in ["happy", "surprise"]:
                emotion_label = "confident"
            elif dominant_emotion in ["neutral"]:
                emotion_label = "neutral"
            else:
                emotion_label = "uncertain"

        finally:
            try: os.remove(video_path)
            except: pass

    for p in (audio_webm, audio_wav):
        try: os.remove(p)
        except: pass

    return jsonify({
        "transcription": transcription,
        "confidence_label": audio_label,  # from audio model
        "emotion": emotion_label          # from video model (mapped)
    })

@app.route("/api/followup_question", methods=["POST"])
def followup_question():
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")

    if not question or not answer:
        return jsonify({"error": "Missing question or answer."}), 400

    prompt = f"""
    You are a technical interviewer. 
    Given the following question and candidate's answer, generate ONE relevant follow-up question.
    Keep it short, clear, and probing deeper.

    Original Question: {question}
    Candidate Answer: {answer}

    Output Format:
    {{
      "followup_question": "..."
    }}
    """

    try:
        raw = get_gemini_response(prompt)
      
      

        return jsonify(raw)

    except Exception as e:
        return jsonify({"error": f"Gemini error: {e}"}), 500


@app.route("/api/video_recommendations", methods=["POST"])
def video_recommendations():
    data = request.json
    user_tags = data.get("tags", [])
    if not user_tags:
        return jsonify({"error": "No tags provided"}), 400

    recommendations = {}

    for tag in user_tags:
        matched_videos = list(
            youtube_collection.find(
                {"metadata_tags": {"$regex": f"{tag}", "$options": "i"}}
            )
        )

        if matched_videos:
            best_video = max(matched_videos, key=lambda x: x.get("llm_score", 0))
            recommendations[tag] = {
                "ID": best_video["ID"],
                "Title": best_video["Title"],
                "llm_score": best_video["llm_score"],
                "source": "db"
            }
        else:
            try:
                url = (
                    f"https://www.googleapis.com/youtube/v3/search?"
                    f"part=snippet&type=video&videoEmbeddable=true&maxResults=5"
                    f"&q={tag}&key={YOUTUBE_API}"
                )
                resp = requests.get(url)
                resp_json = resp.json()
                items = resp_json.get("items", [])
                videos = [{"ID": item["id"]["videoId"], "Title": item["snippet"]["title"]} for item in items]
                recommendations[tag] = {
                    "videos": videos,
                    "source": "youtube"
                }
            except Exception as e:
                recommendations[tag] = {"error": str(e)}

    return jsonify(recommendations)



if __name__ == "__main__":
    app.run(debug=True)
