
"use strict";

// ========================================
// BASIR - Frontend JavaScript
// ========================================

// Elements
const imageInput = document.getElementById("image");
const previewArea = document.getElementById("previewArea");
const preview = document.getElementById("preview");
const imageName = document.getElementById("imageName");

const form = document.getElementById("vqaForm");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("askButton");

const answer = document.getElementById("answer");
const answerActions = document.getElementById("answerActions");
const readAnswerBtn = document.getElementById("readAnswerBtn");
const stopReadingBtn = document.getElementById("stopReadingBtn");
const anotherQuestionBtn = document.getElementById("anotherQuestionBtn");
const statusMessage = document.getElementById("statusMessage");

const openCameraBtn = document.getElementById("openCameraBtn");
const cameraArea = document.getElementById("cameraArea");
const cameraVideo = document.getElementById("cameraVideo");
const capturePhotoBtn = document.getElementById("capturePhotoBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");

const voiceButton = document.getElementById("voiceButton");
const voiceButtonText = document.getElementById("voiceButtonText");
const voiceStatus = document.getElementById("voiceStatus");

const languageToggle = document.getElementById("languageToggle");

// State
let selectedImage = null;
let previewUrl = null;
let cameraStream = null;
let recognition = null;
let isListening = false;
let currentLanguage = "en";
let answerIsDemo = false;
let isSpeaking = false;

const speechSupported = "speechSynthesis" in window;
const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

// ========================================
// Translations
// ========================================

const translations = {
    en: {
        pageTitle: "Basir | Visual Question Answering",
        skip: "Skip to main content",
        home: "Basir Home",
        subtitle: "Visual Question Answering",
        description:
            "Upload or capture an image, enter a question, and receive an answer about its visual content.",
        workspace: "Image Question Answering",
        chooseImage: "Choose an Image",
        upload: "Upload Image",
        takePhoto: "Take Photo",
        imageHelp: "Select a JPG or PNG image, or use your device camera.",
        cameraPreview: "Live camera preview",
        capture: "Capture Photo",
        cancel: "Cancel",
        questionLabel: "Enter Your Question",
        questionPlaceholder: "Type your question about the image.",
        voice: "Ask by Voice",
        stopVoice: "Stop Listening",
        voiceHelp: "You can speak your question in English.",
        submit: "Submit Question",
        answerTitle: "Answer",
        answerPlaceholder: "The answer will be displayed here.",
        read: "Read Answer",
        stopRead: "Stop Reading",
        another: "Ask Another Question",
        team: "Project Team",
        selected: "Selected image: ",
        invalidType: "Please select a JPG or PNG image.",
        invalidAnswer: "Invalid image format.",
        imageSelected: "Image selected.",
        cameraUnavailable: "Camera access is unavailable. Please use localhost or HTTPS.",
        cameraReady: "Camera is ready.",
        cameraError: "Unable to start the camera.",
        cameraDenied: "Camera permission was denied.",
        cameraMissing: "No camera was found.",
        cameraNotReady: "Camera is not ready.",
        captureError: "Unable to capture the image.",
        photoSaved: "Photo captured successfully.",
        cameraClosed: "Camera closed.",
        noImage: "Please select an image before submitting.",
        noImageNotice: "Please upload or capture an image first.",
        processing: "Processing your question...",
        processingNotice: "Processing your question.",
        demo:
            "Your image and question have been received. This is a demonstration. The question answering service is not connected yet.",
        demoNotice: "Answer ready. ",
        playbackUnavailable: "Speech playback is not supported.",
        playbackError: "Speech playback could not be completed.",
        newQuestion: "Enter a new question. The selected image is retained.",
        voiceUnsupported: "Voice input is not supported in this browser. Please type your question instead.",
        listening: "Listening. Please speak your question.",
        voiceCaptured: "Question captured. Review it before submitting.",
        voiceCapturedNotice: "Question captured successfully.",
        voiceDenied: "Microphone permission was denied.",
        noSpeech: "No speech was detected. Please try again.",
        noMicrophone: "No microphone was found.",
        voiceNetwork: "Speech recognition is unavailable.",
        voiceError: "Voice input could not be completed.",
        voiceEnded: "Listening ended. You can try again.",
        voiceStartError: "Unable to start voice input."
    },
    ar: {
        pageTitle: "بصير | الإجابة عن الأسئلة البصرية",
        skip: "تجاوز إلى المحتوى الرئيسي",
        home: "الصفحة الرئيسية لبصير",
        subtitle: "الإجابة عن الأسئلة البصرية",
        description:
            "ارفع صورة أو التقط صورة بالكاميرا، ثم اكتب سؤالًا واحصل على إجابة عن محتواها.",
        workspace: "اسأل عن محتوى الصورة",
        chooseImage: "اختر صورة",
        upload: "رفع صورة",
        takePhoto: "التقاط صورة",
        imageHelp: "اختر صورة بصيغة JPG أو PNG، أو استخدم كاميرا جهازك.",
        cameraPreview: "معاينة الكاميرا المباشرة",
        capture: "التقاط الصورة",
        cancel: "إلغاء",
        questionLabel: "اكتب سؤالك",
        questionPlaceholder: "اكتب سؤالك عن الصورة هنا.",
        voice: "اسأل بصوتك",
        stopVoice: "إيقاف الاستماع",
        voiceHelp: "يمكنك طرح سؤالك صوتيًا بالعربية.",
        submit: "إرسال السؤال",
        answerTitle: "الإجابة",
        answerPlaceholder: "ستظهر الإجابة هنا.",
        read: "قراءة الإجابة",
        stopRead: "إيقاف القراءة",
        another: "طرح سؤال آخر",
        team: "فريق المشروع",
        selected: "الصورة المحددة: ",
        invalidType: "يرجى اختيار صورة بصيغة JPG أو PNG.",
        invalidAnswer: "صيغة الصورة غير مدعومة.",
        imageSelected: "تم اختيار الصورة.",
        cameraUnavailable: "الكاميرا غير متاحة. افتح الموقع باستخدام localhost أو HTTPS.",
        cameraReady: "الكاميرا جاهزة.",
        cameraError: "تعذر تشغيل الكاميرا.",
        cameraDenied: "تم رفض إذن استخدام الكاميرا.",
        cameraMissing: "لم يتم العثور على كاميرا.",
        cameraNotReady: "الكاميرا غير جاهزة.",
        captureError: "تعذر التقاط الصورة.",
        photoSaved: "تم التقاط الصورة بنجاح.",
        cameraClosed: "تم إغلاق الكاميرا.",
        noImage: "يرجى اختيار صورة قبل إرسال السؤال.",
        noImageNotice: "ارفع صورة أو التقط صورة أولًا.",
        processing: "جارٍ معالجة السؤال...",
        processingNotice: "جارٍ معالجة السؤال.",
        demo:
            "تم استلام الصورة والسؤال. هذه إجابة تجريبية، ولم يتم ربط خدمة الإجابة عن الأسئلة بعد.",
        demoNotice: "الإجابة جاهزة. ",
        playbackUnavailable: "القراءة الصوتية غير مدعومة في هذا المتصفح.",
        playbackError: "تعذرت قراءة الإجابة صوتيًا.",
        newQuestion: "اكتب سؤالًا جديدًا. ستبقى الصورة المحددة.",
        voiceUnsupported: "الإدخال الصوتي غير مدعوم في هذا المتصفح. يمكنك كتابة سؤالك.",
        listening: "جارٍ الاستماع. تفضل بطرح سؤالك.",
        voiceCaptured: "تم تسجيل السؤال. راجعه قبل الإرسال.",
        voiceCapturedNotice: "تم تسجيل السؤال بنجاح.",
        voiceDenied: "تم رفض إذن استخدام الميكروفون.",
        noSpeech: "لم يتم اكتشاف كلام. حاول مرة أخرى.",
        noMicrophone: "لم يتم العثور على ميكروفون.",
        voiceNetwork: "خدمة التعرف على الصوت غير متاحة.",
        voiceError: "تعذر إكمال الإدخال الصوتي.",
        voiceEnded: "انتهى الاستماع. يمكنك المحاولة مجددًا.",
        voiceStartError: "تعذر بدء الإدخال الصوتي."
    }
};

function t(key) {
    return translations[currentLanguage][key];
}

function setText(selector, value) {
    const element = document.querySelector(selector);
    if (element) element.textContent = value;
}

function announce(message) {
    if (!statusMessage) return;

    statusMessage.textContent = "";
    requestAnimationFrame(() => {
        statusMessage.textContent = message;
    });
}

// ========================================
// Language Switching
// ========================================

function applyLanguage(language) {
    currentLanguage = language === "ar" ? "ar" : "en";

    document.documentElement.lang = currentLanguage;
    document.documentElement.dir =
        currentLanguage === "ar" ? "rtl" : "ltr";

    document.title = t("pageTitle");

    setText(".skip-link", t("skip"));
    setText(".logo small", t("subtitle"));
    setText(".hero-subtitle", t("subtitle"));
    setText(".hero-description", t("description"));
    setText("#workspace-title", t("workspace"));
    setText(".image-fieldset legend", t("chooseImage"));
    setText(".image-option label[for='image']", "▧ " + t("upload"));
    setText("#openCameraBtn", "◎ " + t("takePhoto"));
    setText("#imageHelp", t("imageHelp"));
    setText("#capturePhotoBtn", t("capture"));
    setText("#closeCameraBtn", t("cancel"));
    setText("label[for='question']", t("questionLabel"));
    setText("#askButton", t("submit"));
    setText("#answer-title", t("answerTitle"));
    setText("#readAnswerBtn", t("read"));
    setText("#stopReadingBtn", t("stopRead"));
    setText("#anotherQuestionBtn", t("another"));
    setText("#team-title", t("team"));

    const logo = document.querySelector(".logo");
    if (logo) logo.setAttribute("aria-label", t("home"));

    cameraVideo.setAttribute("aria-label", t("cameraPreview"));

    questionInput.placeholder = t("questionPlaceholder");

    languageToggle.textContent =
        currentLanguage === "ar" ? "English" : "العربية";

    languageToggle.setAttribute(
        "aria-label",
        currentLanguage === "ar"
            ? "Switch to English"
            : "التبديل إلى العربية"
    );

    languageToggle.lang =
        currentLanguage === "ar" ? "en" : "ar";

    languageToggle.dir =
        currentLanguage === "ar" ? "ltr" : "rtl";

    if (selectedImage) {
        imageName.textContent = t("selected") + selectedImage.name;
    }

    if (answerIsDemo) {
        answer.textContent = t("demo");
    } else if (
        !answerActions.hidden &&
        answer.textContent.trim()
    ) {
        // Keep any actual answer text unchanged.
    } else {
        answer.textContent = t("answerPlaceholder");
    }

    answer.lang = currentLanguage;
    answer.dir = currentLanguage === "ar" ? "rtl" : "ltr";

    if (!SpeechRecognition) {
        voiceStatus.textContent = t("voiceUnsupported");
    } else if (isListening) {
        voiceStatus.textContent = t("listening");
    } else {
        voiceStatus.textContent = t("voiceHelp");
    }

    updateVoiceButton(isListening);

    if (recognition) {
        recognition.lang =
            currentLanguage === "ar" ? "ar-SA" : "en-US";
    }

    if (isSpeaking) stopReading();
}

languageToggle.addEventListener("click", () => {
    applyLanguage(currentLanguage === "en" ? "ar" : "en");
});

// ========================================
// Answer Functions
// ========================================

function stopReading() {
    if (speechSupported) {
        window.speechSynthesis.cancel();
    }

    isSpeaking = false;
    stopReadingBtn.hidden = true;
    readAnswerBtn.hidden = false;
}

function resetAnswer() {
    stopReading();
    answer.textContent = t("answerPlaceholder");
    answer.lang = currentLanguage;
    answerActions.hidden = true;
    answerIsDemo = false;
}

// ========================================
// Image Upload
// ========================================

function handleImage(file) {
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png"];

    if (!allowedTypes.includes(file.type)) {
        announce(t("invalidType"));
        answer.textContent = t("invalidAnswer");
        imageInput.value = "";
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    selectedImage = file;
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    previewArea.hidden = false;
    imageName.textContent = t("selected") + file.name;

    resetAnswer();
    announce(t("imageSelected"));
}

imageInput.addEventListener("change", () => {
    handleImage(imageInput.files[0]);
});

// ========================================
// Camera
// ========================================

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    cameraVideo.srcObject = null;
    cameraArea.hidden = true;
}

async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        announce(t("cameraUnavailable"));
        alert(t("cameraUnavailable"));
        return;
    }

    openCameraBtn.disabled = true;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: { ideal: "environment" }
            }
        });

        cameraVideo.srcObject = cameraStream;
        cameraArea.hidden = false;

        await cameraVideo.play();

        announce(t("cameraReady"));
        capturePhotoBtn.focus();

    } catch (error) {
        stopCamera();

        let message = t("cameraError");

        if (error.name === "NotAllowedError") {
            message = t("cameraDenied");
        } else if (error.name === "NotFoundError") {
            message = t("cameraMissing");
        }

        announce(message);
        alert(message);

    } finally {
        openCameraBtn.disabled = false;
    }
}

function capturePhoto() {
    if (!cameraStream || !cameraVideo.videoWidth) {
        announce(t("cameraNotReady"));
        return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = cameraVideo.videoWidth;
    canvas.height = cameraVideo.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
        announce(t("captureError"));
        return;
    }

    context.drawImage(
        cameraVideo,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(blob => {
        if (!blob) {
            announce(t("captureError"));
            return;
        }

        const photo = new File(
            [blob],
            "basir-camera-photo.jpg",
            { type: "image/jpeg" }
        );

        handleImage(photo);
        stopCamera();

        announce(t("photoSaved"));
        questionInput.focus();

    }, "image/jpeg", 0.9);
}

openCameraBtn.addEventListener("click", openCamera);
capturePhotoBtn.addEventListener("click", capturePhoto);

closeCameraBtn.addEventListener("click", () => {
    stopCamera();
    announce(t("cameraClosed"));
    openCameraBtn.focus();
});

// ========================================
// Submit Question
// ========================================

form.addEventListener("submit", event => {
    event.preventDefault();

    if (!selectedImage) {
        answer.textContent = t("noImage");
        announce(t("noImageNotice"));
        imageInput.focus();
        return;
    }

    if (!form.reportValidity()) return;

    stopReading();

    submitButton.disabled = true;
    answer.textContent = t("processing");
    announce(t("processingNotice"));

    // Temporary response until backend integration.
    const demoAnswer = t("demo");

    answer.textContent = demoAnswer;
    answer.lang = currentLanguage;
    answer.dir = currentLanguage === "ar" ? "rtl" : "ltr";

    answerIsDemo = true;
    answerActions.hidden = false;

    if (!speechSupported) {
        readAnswerBtn.disabled = true;
        readAnswerBtn.title = t("playbackUnavailable");
    }

    submitButton.disabled = false;

    // Announce the result to screen readers.
    announce(t("demoNotice") + demoAnswer);
});

// ========================================
// Text-to-Speech
// ========================================

readAnswerBtn.addEventListener("click", () => {
    if (!speechSupported) {
        announce(t("playbackUnavailable"));
        return;
    }

    stopReading();

    const utterance = new SpeechSynthesisUtterance(
        answer.textContent
    );

    utterance.lang =
        answer.lang === "ar" ? "ar-SA" : "en-US";

    utterance.rate = 0.9;
    utterance.pitch = 1;

    utterance.onstart = () => {
        isSpeaking = true;
    };

    utterance.onend = () => {
        isSpeaking = false;
        stopReadingBtn.hidden = true;
        readAnswerBtn.hidden = false;
    };

    utterance.onerror = event => {
        isSpeaking = false;
        stopReadingBtn.hidden = true;
        readAnswerBtn.hidden = false;

        if (event.error !== "canceled") {
            announce(t("playbackError"));
        }
    };

    readAnswerBtn.hidden = true;
    stopReadingBtn.hidden = false;

    window.speechSynthesis.speak(utterance);
});

stopReadingBtn.addEventListener("click", () => {
    stopReading();
    readAnswerBtn.focus();
});

// ========================================
// Ask Another Question
// ========================================

anotherQuestionBtn.addEventListener("click", () => {
    questionInput.value = "";
    resetAnswer();
    announce(t("newQuestion"));
    questionInput.focus();
});

// ========================================
// Voice Input - Speech-to-Text
// ========================================

function updateVoiceButton(listening) {
    isListening = listening;

    voiceButton.classList.toggle("is-listening", listening);
    voiceButton.setAttribute(
        "aria-pressed",
        String(listening)
    );

    voiceButtonText.textContent =
        listening ? t("stopVoice") : t("voice");
}

if (!SpeechRecognition) {
    voiceButton.disabled = true;
    voiceStatus.textContent = t("voiceUnsupported");

} else {
    recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        updateVoiceButton(true);
        voiceStatus.textContent = t("listening");
    };

    recognition.onresult = event => {
        const transcript =
            event.results[0][0].transcript.trim();

        questionInput.value = transcript;

        voiceStatus.textContent = t("voiceCaptured");
        announce(t("voiceCapturedNotice"));
    };

    recognition.onerror = event => {
        const messages = {
            "not-allowed": t("voiceDenied"),
            "no-speech": t("noSpeech"),
            "audio-capture": t("noMicrophone"),
            "network": t("voiceNetwork")
        };

        voiceStatus.textContent =
            messages[event.error] || t("voiceError");

        updateVoiceButton(false);
    };

    recognition.onend = () => {
        updateVoiceButton(false);

        if (voiceStatus.textContent === t("listening")) {
            voiceStatus.textContent = t("voiceEnded");
        }
    };

    voiceButton.addEventListener("click", () => {
        if (isListening) {
            recognition.stop();
        } else {
            try {
                recognition.lang =
                    currentLanguage === "ar"
                        ? "ar-SA"
                        : "en-US";

                recognition.start();
            } catch (error) {
                voiceStatus.textContent = t("voiceStartError");
                console.error("Voice input error:", error);
            }
        }
    });
}

// ========================================
// Cleanup
// ========================================

window.addEventListener("pagehide", () => {
    stopCamera();
    stopReading();

    if (recognition && isListening) {
        recognition.abort();
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
});

// Initialize English interface.
applyLanguage("en");
