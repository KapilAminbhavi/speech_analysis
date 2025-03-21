import streamlit as st
import tempfile
import os
import json
import whisper
import pronouncing
from openai import OpenAI
import base64
import librosa

# Set page config
st.set_page_config(
    page_title="English Pronunciation Analyzer",
    page_icon="🎙️",
    layout="wide"
)

import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in Streamlit Cloud secrets
if not api_key:
    raise ValueError("API key is missing!")
client = OpenAI(api_key=api_key)

# Load Whisper model (cache it)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Transcribe audio with Whisper
def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path)
    return result["text"]

# Phonetic analysis
def analyze_phonetics(text):
    words = text.lower().split()
    phoneme_analysis = {}
    for word in words:
        clean_word = ''.join(e for e in word if e.isalnum())
        if clean_word:
            phonemes = pronouncing.phones_for_word(clean_word)
            phoneme_analysis[clean_word] = phonemes if phonemes else "No phoneme found"
    return phoneme_analysis

# Analyze speech with GPT
def analyze_speech(text, phoneme_data, audio_duration=None, word_count=None, language_hint=None, expected_topic=None):
    """Use GPT to analyze pronunciation, grammar & MTI with comprehensive metrics"""

    # Calculate speaking rate if available
    speaking_rate = None
    if audio_duration and word_count:
        speaking_rate = round(word_count / (audio_duration / 60))

    language_hint_text = f"The speaker's likely native language is {language_hint}." if language_hint else ""
    expected_topic_text = f"Expected speaking topic: {expected_topic}" if expected_topic else "No specific topic was provided for relevance assessment."

    prompt = f"""
    # Comprehensive English Speech Analysis

    Please analyze this transcribed English speech thoroughly using the provided data and create a detailed assessment report.

    ## Transcribed Speech:
    "{text}"

    ## Phonetic Analysis:
    {json.dumps(phoneme_data, indent=2)}

    ## Additional Information:
    {f"- Speaking rate: {speaking_rate} words per minute" if speaking_rate else ""}
    {language_hint_text}
    {expected_topic_text}

    # Analysis Requirements

    ## 1. CEFR Level Assessment
    Rate the speaker on the CEFR scale (A1-C2) for each of the following categories:
    - Fluency
    - Grammar
    - Vocabulary
    - Pronunciation
    - Interaction

    Example assessment:
    ```
    Fluency: C1 - Can express themselves fluently and spontaneously without much obvious searching for expressions.
    Grammar: C2 - Maintains consistent grammatical control of complex language with minimal errors.
    Vocabulary: B2 - Has a good range of vocabulary for matters connected to their field and most general topics.
    Pronunciation: C2 - Has acquired a clear, natural pronunciation and intonation.
    Interaction: C1 - Can select a suitable phrase from a readily available range of discourse functions to preface remarks.
    Overall CEFR Level: C1 (Advanced)
    ```

    ## 2. Strengths & Areas for Improvement
    Identify clear strengths and specific areas for improvement.

    Example:
    ```
    Strengths:
    - You made very few grammatical errors.
    - You used 5 advanced grammar constructions (relative clauses, passive voice, tags, etc.)
    - 75% of your sentences had a complex structure.
    - You're an expert at using phrasal verbs ("get along", "have been", etc.)
    - You have a large active vocabulary with high-level words.

    Areas for Improvement:
    - Your speaking rate (99 wpm) and pausing are below native level (90-150 wpm).
    - Reduce the number of time-fillers: "also" 8 times, "always" 6 times, "like" 5 times.
    - Use more synonyms for frequently repeated words.
    - Add various linking words (such as "besides", "in order to", "therefore").
    ```

    ## 3. Native-Like Rephrasing
    Provide examples of how a native speaker would rephrase 5 of the speaker's sentences.

    Example:
    ```
    Original: "My name is Joy and I have a degree in Bachelor of Secondary Education."
    Native-like: "Hi, I'm Joy and I graduated with a degree in Bachelor of Secondary Education."

    Original: "I have five years of teaching experience and also have other jobs like hosting events."
    Native-like: "I've been teaching for five years and I also have experience hosting events."
    ```

    ## 4. Vocabulary Analysis
    Analyze the speaker's vocabulary usage with statistics.

    Example:
    ```
    Vocabulary Statistics:
    - Active vocabulary: approximately 3688 words (B2 level)
    - Unique words: 128 words used only once in your speech
    - Rare words: 30% of words used are not among the 5,000 most common English words
    - Common words: 84% of words used are among the 2,000 most frequently used English words

    Vocabulary Level Distribution:
    - Beginner (A1): 30%
    - Elementary (A2): 19%
    - Intermediate (B1): 27%
    - Upper-intermediate (B2): 20%
    - Advanced (C1): 2%
    - Proficiency (C2): 2%
    ```

    ## 5. Word Level Classification
    Provide examples of words used at different CEFR levels.

    Example:
    ```
    A1 words: different, difficult, sometimes, birthday
    A2 words: first of all, text message, as well as, be able to
    B1 words: communication, experienced, right after, at same time
    B2 words: get involved with, emotionally, aside from, commitment
    C1 words: ministry, outcome, days
    C2 words: have no time for, be something, dated
    ```

    ## 6. Word Repetition Analysis
    Identify the most frequently repeated words or phrases.

    Example:
    ```
    Top repeated words:
    - "also" - 8 times
    - "enjoy" - 7 times
    - "online conference" - 5 times
    - "always" - 4 times
    - "job seeker communicate" - 3 times

    Suggestion: Replace some instances with alternatives such as:
    - instead of "also" try: "additionally", "furthermore", "moreover", "in addition"
    - instead of "enjoy" try: "appreciate", "take pleasure in", "find satisfying", "delight in"
    ```

    ## 7. Speaking Rate Analysis
    Analyze the speaker's pace and fluency.

    Example:
    ```
    Speaking rate: 99 words per minute
    - This is within the lower end of the normal range for native English speakers (90-150 wpm)
    - A slightly increased rate might make your speech more engaging
    - Your current rate may be perceived as somewhat measured or careful

    Pausing patterns:
    - You pause frequently between sentences (good for clarity)
    - Some pauses within sentences disrupt natural flow
    - Suggestion: Work on reducing mid-sentence pauses except when emphasizing key points
    ```

    ## 8. Pronunciation Analysis
    Identify specific pronunciation issues and patterns.

    Example:
    ```
    Pronunciation patterns:
    - Consistent difficulty with "th" sounds (pronouncing them as "d" or "z")
    - Tendency to reduce vowel contrast between "ship" and "sheep"
    - Word stress often placed on incorrect syllables (e.g., "CON-tent" instead of "con-TENT")
    - Difficulty with consonant clusters in words like "strengths" or "sixths"

    Phonetic inaccuracies that change meaning:
    - "sheet" pronounced as "shit" (vowel length issue)
    - "peace" pronounced similar to "piss" (vowel quality)
    - "focus" pronounced similar to "fox" (syllable reduction)
    ```

    ## Mother Tongue Influence (MTI)
    Identify specific patterns that suggest influence from the speaker's native language.

    ## Grammar Analysis
    Identify grammar patterns, both strengths and errors.

    ## Specific Improvement Plan
    Provide 3-5 concrete, actionable tips for improvement.

    ## 9. Topic Relevance Analysis
    Analyze how relevant the speech is to the expected topic.

    Example:
    ```
    Expected Topic: "Car"
    Relevance Score: 3/10 (Low)

    Topic Analysis:
    - Only 15% of the content was related to cars or transportation
    - The speaker frequently digressed to unrelated topics such as weather, food, and personal hobbies
    - Main irrelevant topics mentioned: sun (5 times), cooking (3 times), weekend activities (2 minutes)
    - Recommendation: Focus more directly on the assigned topic and avoid unnecessary digressions
    ```

    Please include this relevance analysis in your report. If no expected topic was provided, note that topic relevance couldn't be assessed.

    Please format your response with clear sections and bullet points for each category.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Main app
def main():
    st.title("🎙️ English Pronunciation Analyzer")
    st.markdown("### Analyze your spoken English for pronunciation, grammar, and mother tongue influence")

    # Load Whisper model
    with st.spinner("Loading speech recognition model..."):
        model = load_whisper_model()

    # Sidebar options
    st.sidebar.title("Options")
    language_hint = st.sidebar.text_input("Your native language (optional):", help="Improves analysis accuracy")
    expected_topic = st.sidebar.text_input("Expected topic (optional):", help="Topic you should discuss")

    # Initialize session state
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = ''
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ''

    # Recording section
    st.markdown("### 🎤 Record Your Speech")
    st.markdown(
        "Click 'Start Recording' to begin speaking. Speak as little or as much as you like, then click 'Stop Recording'. Edit the transcription below if needed, then click 'Analyze Speech'.")

    
    
    # Display topic
    topic = "The importance of learning English"
    st.info(f"Your topic is: {topic}")

    # HTML and JavaScript for recording and real-time transcription
    recording_html = """
    <div>
        <button type="button" id="start-btn">Start Recording</button>
        <button type="button" id="stop-btn" disabled>Stop Recording</button>
        <textarea id="transcription" style="width:100%; height:200px; margin-top:10px;" placeholder="Your speech will appear here..."></textarea>
        <input type="hidden" id="audio_data" name="audio_data">
    </div>
    <script>
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        const transcriptionArea = document.getElementById('transcription');
        const audioDataInput = document.getElementById('audio_data');

        let recognition;
        let mediaRecorder;
        let audioChunks = [];

        // Speech recognition setup
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                transcriptionArea.value = transcript;
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                transcriptionArea.value = '[Error: ' + event.error + ']';
            };
        } else {
            alert('Speech recognition not supported in this browser.');
            transcriptionArea.value = 'Speech recognition not supported.';
        }

        // Audio recording setup
        navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    audioDataInput.value = base64Audio;
                    window.parent.postMessage({
                        type: 'streamlit:set_component_value',
                        value: {audio_data: base64Audio, transcription: transcriptionArea.value}
                    }, '*');
                };
                audioChunks = [];
            };
        }).catch(err => {
            console.error('Microphone access error:', err);
            alert('Could not access microphone: ' + err.message);
        });

        // Button event listeners
        startBtn.addEventListener('click', () => {
            if (recognition && mediaRecorder) {
                transcriptionArea.value = '';
                recognition.start();
                mediaRecorder.start();
                startBtn.disabled = true;
                stopBtn.disabled = false;
            }
        });

        stopBtn.addEventListener('click', () => {
            if (recognition && mediaRecorder) {
                recognition.stop();
                mediaRecorder.stop();
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        });
    </script>
    """

    # Render the HTML component
    component_value = st.components.v1.html(recording_html, height=300)

    # Update session state with component value
    if component_value and isinstance(component_value, dict):
        st.session_state.audio_data = component_value.get('audio_data', '')
        st.session_state.transcription = component_value.get('transcription', '')

    st.markdown("Please copy and paste the above text into the box below for the analysis")
    
    # Form for submission with editable transcription
    with st.form(key='recording_form'):
        # Editable transcription area
        transcription_input = st.text_area(
            "Edit your transcription here:",
            value=st.session_state.transcription,
            height=200,
            key="transcription_input"
        )
        submit_button = st.form_submit_button("Analyze Speech")

    # Process submitted audio
    if submit_button:
        audio_data = st.session_state.audio_data
        transcription = transcription_input  # Use the edited text from the form

        if transcription or audio_data:  # Check if either is present
            with st.spinner("Analyzing your speech..."):
                try:
                    if audio_data:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(audio_data)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                            temp_audio.write(audio_bytes)
                            temp_audio_path = temp_audio.name

                        # Use Whisper only if no transcription provided
                        final_transcription = transcription if transcription else transcribe_audio(temp_audio_path, model)
                        audio_duration = librosa.get_duration(path=temp_audio_path)
                        word_count = len(final_transcription.split())
                    else:
                        final_transcription = transcription
                        audio_duration = None
                        word_count = len(final_transcription.split()) if final_transcription else 0

                    # Analyze phonemes
                    phoneme_data = analyze_phonetics(final_transcription)

                    # Analyze with OpenAI
                    analysis_results = analyze_speech(
                        final_transcription,
                        phoneme_data,
                        audio_duration=audio_duration,
                        word_count=word_count,
                        language_hint=language_hint,
                        expected_topic=topic
                    )

                    # Display results
                    st.markdown("## 📊 Analysis Results")
                    st.markdown("### Your Transcription:")
                    st.markdown(
                        f'<div style="background-color: #e1f5fe; padding: 15px; border-radius: 10px;">{final_transcription}</div>',
                        unsafe_allow_html=True)

                    if audio_duration:
                        st.markdown("### Audio Metrics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Duration", f"{audio_duration:.2f} seconds")
                        with col2:
                            speaking_rate = round(word_count / (audio_duration / 60))
                            st.metric("Speaking Rate", f"{speaking_rate} words/minute")

                    with st.expander("Show Phonetic Analysis"):
                        st.json(phoneme_data)

                    st.markdown("### Comprehensive Speech Analysis")
                    st.markdown(analysis_results)

                    # Clean up
                    if 'temp_audio_path' in locals():
                        os.remove(temp_audio_path)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    if 'temp_audio_path' in locals():
                        os.remove(temp_audio_path)
        else:
            st.error("No audio or transcription provided. Please record or enter text before analyzing.")

    # File upload section
    st.markdown("---")
    st.markdown("### 🎤 Or Upload an Audio File")
    audio_file = st.file_uploader("Upload your audio file (WAV/MP3)", type=["wav", "mp3"])
    if audio_file is not None:
        with st.spinner("Analyzing your speech..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name
            try:
                transcribed_text = transcribe_audio(temp_audio_path, model)
                word_count = len(transcribed_text.split())
                audio_duration = librosa.get_duration(path=temp_audio_path)
                phoneme_data = analyze_phonetics(transcribed_text)
                analysis_results = analyze_speech(
                    transcribed_text,
                    phoneme_data,
                    audio_duration=audio_duration,
                    word_count=word_count,
                    language_hint=language_hint,
                    expected_topic=expected_topic
                )
                st.markdown("## 📊 Analysis Results")
                st.markdown("### Transcribed Speech:")
                st.markdown(
                    f'<div style="background-color: #e1f5fe; padding: 15px; border-radius: 10px;">{transcribed_text}</div>',
                    unsafe_allow_html=True)
                if audio_duration:
                    st.markdown("### Audio Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{audio_duration:.2f} seconds")
                    with col2:
                        speaking_rate = round(word_count / (audio_duration / 60))
                        st.metric("Speaking Rate", f"{speaking_rate} words/minute")
                with st.expander("Show Phonetic Analysis"):
                    st.json(phoneme_data)
                st.markdown("### Comprehensive Speech Analysis")
                st.markdown(analysis_results)
                os.remove(temp_audio_path)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
