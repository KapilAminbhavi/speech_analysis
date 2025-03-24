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
    # Comprehensive English Speech Analysis Protocol
    
    Analyze the provided English speech transcript according to these precise guidelines to generate a standardized assessment report.
    
    ## Input Data
    
    - **Speech Transcript:** "{text}"
    - **Phonetic Data:** {json.dumps(phoneme_data, indent=2)}
    - **Speaking Rate:** {speaking_rate} words per minute (if available)
    - **Language Background:** {language_hint_text}
    - **Expected Topic:** {expected_topic_text}
    
    ## Required Analysis Sections
    
    ### 1. CEFR Level Assessment
    
    Provide a precise CEFR rating (A1-C2) for each category with supporting evidence:
    
    | Category | Rating | Evidence |
    |----------|--------|----------|
    | Fluency | [CEFR Level] | [Specific evidence from transcript] |
    | Grammar | [CEFR Level] | [Specific evidence from transcript] |
    | Vocabulary | [CEFR Level] | [Specific evidence from transcript] |
    | Pronunciation | [CEFR Level] | [Specific evidence from transcript] |
    | Interaction | [CEFR Level] | [Specific evidence from transcript] |
    | **Overall CEFR Level** | [CEFR Level] | [Summary justification] |
    
    ### 2. Quantitative Strengths & Weaknesses Analysis
    
    **Strengths:**
    - **Grammar Accuracy:** [X]% of sentences grammatically correct
    - **Sentence Complexity:** [X]% complex sentences, [X]% compound sentences, [X]% simple sentences
    - **Advanced Constructions:** [Exact number] of [specific constructions] used
    - **Vocabulary Range:** [Specific metrics about vocabulary diversity]
    
    **Areas for Improvement:**
    - **Speaking Rate:** [Specific comparison to target range with percentage deviation]
    - **Filler Usage:** [Exact count] of each filler word/phrase
    - **Word Repetition:** [Specific words/phrases with exact repetition counts]
    - **Linking Words:** [Analysis of connector usage with specific metrics]
    
    ### 3. Native-Like Rephrasing
    
    Identify exactly 5 non-native-like constructions and provide native-like alternatives:
    
    | Non-Native Construction | Native-Like Alternative | Improvement Explanation |
    |-------------------------|-------------------------|-------------------------|
    | [Direct quote from transcript] | [Improved version] | [Specific linguistic explanation] |
    | [Direct quote from transcript] | [Improved version] | [Specific linguistic explanation] |
    | [Direct quote from transcript] | [Improved version] | [Specific linguistic explanation] |
    | [Direct quote from transcript] | [Improved version] | [Specific linguistic explanation] |
    | [Direct quote from transcript] | [Improved version] | [Specific linguistic explanation] |
    
    ### 4. Vocabulary Metrics
    
    **Quantitative Analysis:**
    - **Total Word Count:** [Exact number]
    - **Unique Word Count:** [Exact number]
    - **Type-Token Ratio:** [Calculated ratio]
    - **Lexical Density:** [Calculated percentage]
    - **Academic Word List Coverage:** [Percentage of academic vocabulary]
    
    **CEFR Level Distribution:**
    - A1: [X]% ([number] words)
    - A2: [X]% ([number] words)
    - B1: [X]% ([number] words)
    - B2: [X]% ([number] words)
    - C1: [X]% ([number] words)
    - C2: [X]% ([number] words)
    
    ### 5. Word Level Classification Table
    
    Provide exactly 5 examples for each CEFR level:
    
    | CEFR Level | Word/Phrase Examples |
    |------------|----------------------|
    | A1 | [5 specific examples from transcript] |
    | A2 | [5 specific examples from transcript] |
    | B1 | [5 specific examples from transcript] |
    | B2 | [5 specific examples from transcript] |
    | C1 | [5 specific examples from transcript] |
    | C2 | [5 specific examples from transcript] |
    
    ### 6. Word Repetition Analysis
    
    **High-Frequency Words/Phrases:**
    - [Word/phrase]: [Exact count] occurrences
    - [Word/phrase]: [Exact count] occurrences
    - [Word/phrase]: [Exact count] occurrences
    - [Word/phrase]: [Exact count] occurrences
    - [Word/phrase]: [Exact count] occurrences
    
    **Suggested Alternatives:**
    - For [word/phrase]: [alternative 1], [alternative 2], [alternative 3]
    - For [word/phrase]: [alternative 1], [alternative 2], [alternative 3]
    
    ### 7. Speaking Rate and Pause Analysis
    
    **Rate Metrics:**
    - **Words Per Minute:** [Exact number]
    - **Syllables Per Minute:** [Estimated number]
    - **Comparison to Native Range:** [Deviation percentage]
    
    **Pause Analysis:**
    - **Total Pauses:** [Exact number]
    - **Natural Pauses:** [Exact number] ([X]%)
    - **Hesitation Pauses:** [Exact number] ([X]%)
    - **Pause Frequency:** 1 pause every [X] words
    - **Average Pause Duration:** [Estimated duration]
    
    **Fluency Assessment:**
    - **Pause-to-Speech Ratio:** [Calculated ratio]
    - **Flow Disruption Score:** [Quantitative measure]
    - **Specific Disruption Patterns:** [List of patterns with examples]
    
    ### 8. Pronunciation Analysis
    
    **Phoneme Accuracy:**
    - **Vowels:** [X]% accuracy
    - **Consonants:** [X]% accuracy
    - **Consonant Clusters:** [X]% accuracy
    - **Diphthongs:** [X]% accuracy
    
    **Specific Issues:**
    - [Phoneme]: [Description of issue] in words [example 1], [example 2]
    - [Phoneme]: [Description of issue] in words [example 1], [example 2]
    
    **Prosodic Features:**
    - **Word Stress:** [Analysis with specific examples]
    - **Sentence Stress:** [Analysis with specific examples]
    - **Intonation Patterns:** [Analysis with specific examples]
    - **Rhythm:** [Analysis with specific examples]
    
    ### 9. Mother Tongue Influence (MTI)
    
    - **Identified First Language:** [Language] (confidence level: [high/medium/low])
    - **Phonological Influences:** [Specific patterns with examples]
    - **Syntactic Influences:** [Specific patterns with examples]
    - **Lexical Influences:** [Specific patterns with examples]
    
    ### 10. Grammar Analysis
    
    **Accuracy Statistics:**
    - **Error-Free Clauses:** [X]%
    - **Error-Free T-Units:** [X]%
    
    **Grammar Strengths:**
    - [Specific construction]: Used correctly [X] times
    - [Specific construction]: Used correctly [X] times
    
    **Grammar Errors:**
    - [Error type]: [X] instances (e.g., [example from transcript])
    - [Error type]: [X] instances (e.g., [example from transcript])
    
    ### 11. Improvement Plan
    
    | Priority | Focus Area | Specific Exercise | Expected Outcome | Time Frame |
    |----------|------------|-------------------|------------------|------------|
    | 1 | [Area] | [Detailed exercise] | [Measurable outcome] | [Duration] |
    | 2 | [Area] | [Detailed exercise] | [Measurable outcome] | [Duration] |
    | 3 | [Area] | [Detailed exercise] | [Measurable outcome] | [Duration] |
    | 4 | [Area] | [Detailed exercise] | [Measurable outcome] | [Duration] |
    | 5 | [Area] | [Detailed exercise] | [Measurable outcome] | [Duration] |
    
    ### 12. Topic Relevance Analysis
    
    - **Expected Topic:** [Topic]
    - **Relevance Score:** [0-10]
    - **On-Topic Content:** [X]%
    - **Off-Topic Content:** [X]%
    - **Main Digressions:** [List with frequency]
    - **Coherence Assessment:** [Analysis of logical flow]
    
    ## Report Format Requirements
    
    1. All sections must be completed with precise quantitative data where applicable
    2. No subjective evaluations without supporting evidence
    3. All examples must be direct quotes from the transcript
    4. Tables must be properly formatted with aligned columns
    5. Numerical data must include units of measurement where appropriate
    6. Analysis must be evidence-based with specific references to the speech content
    7. All improvement recommendations must be actionable and specific

    Please include this relevance analysis in your report. If no expected topic was provided, note that topic relevance couldn't be assessed.

    Please format your response with clear sections and bullet points for each category.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
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
        let currentTranscription = ''; // Store finalized transcription

        // Speech recognition setup
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;

            recognition.onresult = (event) => {
                let interimTranscript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        // Append finalized transcript and add a space
                        currentTranscription += transcript + ' ';
                        transcriptionArea.value = currentTranscription.trim();
                    } else {
                        // Show interim transcript temporarily without appending to final text
                        interimTranscript = transcript;
                        transcriptionArea.value = (currentTranscription + ' ' + interimTranscript).trim();
                    }
                }
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                transcriptionArea.value = currentTranscription + ' [Error: ' + event.error + ']';
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
                // Clear interim but keep finalized text on new recording start
                transcriptionArea.value = currentTranscription;
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
                        final_transcription = transcription if transcription else transcribe_audio(temp_audio_path,
                                                                                                   model)
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
