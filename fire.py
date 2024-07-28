# استيراد المكتبات اللازمة
import cv2  # مكتبة OpenCV للتعامل مع الصور والفيديو
import threading  # مكتبة لتشغيل العمليات في خيوط متعددة
import pygame  # مكتبة للتعامل مع الألعاب والصوت
import math  # مكتبة للعمليات الرياضية
from ultralytics import YOLO  # مكتبة YOLO

# تعريف الكلاس FireDetector
class FireDetector:
    # المُنشئ للكلاس
    def __init__(self, model_path, sound_file):
        # تحميل نموذج YOLO
        self.model = YOLO(model_path)
        # تخزين مسار ملف الصوت
        self.sound_file = sound_file
        # فتح الكاميرا
        self.cap = cv2.VideoCapture(1)
        # تهيئة مكبر الصوت
        pygame.mixer.init()

    # دالة لتشغيل الصوت
    def play_sound(self):
        pygame.mixer.music.load(self.sound_file)
        pygame.mixer.music.play()

    # دالة للكشف عن الحرائق
    def detect_fire(self):
        while True:
            # قراءة الإطار من الكاميرا
            ret, frame = self.cap.read()
            if not ret:
                break

            # عكس اتجاه الكاميرا من اليمين إلى اليسار
            frame = cv2.flip(frame, 1)

            # استخدام نموذج YOLO للكشف عن الحرائق
            results = self.model(frame, stream=True)

            fire_detected = False
            
            # الحصول على معلومات bbox والثقة وأسماء الفئات للعمل معها
            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Fire {confidence}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        fire_detected = True

            # تشغيل صوت الإنذار في خيط جديد إذا تم اكتشاف النار
            if fire_detected:
                if not pygame.mixer.music.get_busy():
                    threading.Thread(target=self.play_sound).start()
            else:
                # إذا لم يتم الكشف عن الحرائق، إيقاف الصوت فورًا
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

            # عرض الإطار
            cv2.imshow('Fire Detection', frame)

            # إذا تم الضغط على مفتاح الخروج (ESC)، إنهاء الحلقة
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # إغلاق الكاميرا وإغلاق جميع النوافذ
        self.cap.release()
        cv2.destroyAllWindows()

# إنشاء كائن من الكلاس FireDetector
fire_detector = FireDetector('best.pt', 'Alarm Sound.mp3')
# بدء الكشف عن الحرائق
fire_detector.detect_fire()