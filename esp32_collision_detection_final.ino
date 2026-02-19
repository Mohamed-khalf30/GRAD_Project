/*
 * ESP32 Collision Detection System
 * Using TensorFlow Lite for Microcontrollers
 * Threshold: 0.35
 * Ready for IMU sensors (MPU6050, MPU9250, etc.)
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"
#include "preprocessing.h"

// ============ إعدادات النظام ============
#define COLLISION_THRESHOLD 0.35  // عتبة الكشف عن الاصطدام
#define SAMPLING_RATE_MS 10       // 100 Hz sampling rate

// ============ إعدادات الذاكرة ============
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// ============ متغيرات TensorFlow ============
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ============ متغيرات البيانات ============
float data_buffer[WINDOW_SIZE][NUM_FEATURES];
int buffer_index = 0;
unsigned long last_timestamp = 0;

// ============ متغيرات الفلتر (لكل قناة) ============
float filter_x[9][FILTER_ORDER + 1] = {0};
float filter_y[9][FILTER_ORDER + 1] = {0};

// ============ إحصائيات ============
unsigned long total_inferences = 0;
unsigned long total_collisions = 0;
float avg_inference_time = 0;

// ============================================
//              Setup Function
// ============================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n=================================");
  Serial.println("  Collision Detection System");
  Serial.println("  Threshold: 0.35");
  Serial.println("=================================\n");
  
  // ضبط أقصى سرعة للـ CPU
  setCpuFrequencyMhz(240);
  Serial.println("✓ CPU frequency: 240 MHz");
  
  // تحميل النموذج
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("✗ Model schema mismatch!");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Expected version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while(1) delay(1000);
  }
  Serial.println("✓ Model loaded successfully");
  Serial.print("  Model size: ");
  Serial.print(model_data_len);
  Serial.println(" bytes");
  
  // إعداد الـ operations
  static tflite::AllOpsResolver resolver;
  
  // إنشاء الـ interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // تخصيص الذاكرة
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("✗ AllocateTensors() failed");
    while(1) delay(1000);
  }
  Serial.println("✓ Tensors allocated");
  
  // الحصول على pointers
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  // طباعة معلومات الذاكرة
  Serial.print("  Arena used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(kTensorArenaSize);
  Serial.println(" bytes");
  
  Serial.print("  Free heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
  
  // طباعة معلومات النموذج
  Serial.println("\n✓ Model Configuration:");
  Serial.print("  Window size: ");
  Serial.println(WINDOW_SIZE);
  Serial.print("  Features: ");
  Serial.println(NUM_FEATURES);
  Serial.print("  Threshold: ");
  Serial.println(COLLISION_THRESHOLD, 2);
  
  Serial.println("\n=================================");
  Serial.println("System Ready!");
  Serial.println("=================================");
  Serial.println("\nWaiting for IMU data...");
  Serial.println("Format: acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z\n");
  
  // اختبار اختياري
  // test_with_random_data();
}

// ============================================
//       Butterworth Low-Pass Filter
// ============================================
float butterworth_filter(float input_value, int channel) {
  // Shift old values
  for (int i = FILTER_ORDER; i > 0; i--) {
    filter_x[channel][i] = filter_x[channel][i-1];
    filter_y[channel][i] = filter_y[channel][i-1];
  }
  
  filter_x[channel][0] = input_value;
  
  // Apply filter equation
  filter_y[channel][0] = FILTER_B[0] * filter_x[channel][0];
  for (int i = 1; i <= FILTER_ORDER; i++) {
    filter_y[channel][0] += FILTER_B[i] * filter_x[channel][i];
    filter_y[channel][0] -= FILTER_A[i] * filter_y[channel][i];
  }
  
  return filter_y[channel][0];
}

// ============================================
//          Preprocessing Function
// ============================================
void preprocess_and_add_data(float acc_x, float acc_y, float acc_z,
                             float gyro_x, float gyro_y, float gyro_z,
                             float mag_x, float mag_y, float mag_z) {
  
  // حساب delta_t
  unsigned long current_time = millis();
  float delta_t = 0;
  if (last_timestamp > 0) {
    delta_t = (current_time - last_timestamp) / 1000.0; // بالثواني
  }
  last_timestamp = current_time;
  
  // تطبيق Low-pass filter (كل قناة لها filter منفصل)
  acc_x = butterworth_filter(acc_x, 0);
  acc_y = butterworth_filter(acc_y, 1);
  acc_z = butterworth_filter(acc_z, 2);
  gyro_x = butterworth_filter(gyro_x, 3);
  gyro_y = butterworth_filter(gyro_y, 4);
  gyro_z = butterworth_filter(gyro_z, 5);
  mag_x = butterworth_filter(mag_x, 6);
  mag_y = butterworth_filter(mag_y, 7);
  mag_z = butterworth_filter(mag_z, 8);
  
  // Normalize delta_t
  delta_t = delta_t / MAX_DELTA_T;
  if (delta_t > 1.0) delta_t = 1.0;
  
  // إضافة للـ buffer مع Standardization
  data_buffer[buffer_index][0] = (acc_x - SCALER_MEAN[0]) / SCALER_SCALE[0];
  data_buffer[buffer_index][1] = (acc_y - SCALER_MEAN[1]) / SCALER_SCALE[1];
  data_buffer[buffer_index][2] = (acc_z - SCALER_MEAN[2]) / SCALER_SCALE[2];
  data_buffer[buffer_index][3] = (gyro_x - SCALER_MEAN[3]) / SCALER_SCALE[3];
  data_buffer[buffer_index][4] = (gyro_y - SCALER_MEAN[4]) / SCALER_SCALE[4];
  data_buffer[buffer_index][5] = (gyro_z - SCALER_MEAN[5]) / SCALER_SCALE[5];
  data_buffer[buffer_index][6] = (mag_x - SCALER_MEAN[6]) / SCALER_SCALE[6];
  data_buffer[buffer_index][7] = (mag_y - SCALER_MEAN[7]) / SCALER_SCALE[7];
  data_buffer[buffer_index][8] = (mag_z - SCALER_MEAN[8]) / SCALER_SCALE[8];
  data_buffer[buffer_index][9] = delta_t;
  
  buffer_index++;
}

// ============================================
//          Run Inference
// ============================================
bool run_inference() {
  if (buffer_index < WINDOW_SIZE) {
    return false; // لسه مش جاهزين
  }
  
  // نسخ البيانات للـ input tensor
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int j = 0; j < NUM_FEATURES; j++) {
      input_tensor->data.f[i * NUM_FEATURES + j] = data_buffer[i][j];
    }
  }
  
  // تشغيل الـ inference
  unsigned long start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long end = micros();
  unsigned long inference_time = end - start;
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("✗ Invoke failed!");
    return false;
  }
  
  // قراءة النتيجة وتطبيق threshold = 0.35
  float prediction = output_tensor->data.f[0];
  bool collision = (prediction > COLLISION_THRESHOLD);
  
  // تحديث الإحصائيات
  total_inferences++;
  if (collision) total_collisions++;
  avg_inference_time = (avg_inference_time * (total_inferences - 1) + (inference_time / 1000.0)) / total_inferences;
  
  // طباعة النتائج
  Serial.print("[");
  Serial.print(millis());
  Serial.print(" ms] ");
  Serial.print("Prediction: ");
  Serial.print(prediction, 4);
  Serial.print(" | Collision: ");
  if (collision) {
    Serial.print("⚠️  YES  ⚠️");
  } else {
    Serial.print("✓ NO");
  }
  Serial.print(" | Time: ");
  Serial.print(inference_time / 1000.0, 2);
  Serial.print(" ms | Total: ");
  Serial.print(total_collisions);
  Serial.print("/");
  Serial.println(total_inferences);
  
  // تحريك الـ buffer (sliding window)
  for (int i = 0; i < WINDOW_SIZE - 1; i++) {
    for (int j = 0; j < NUM_FEATURES; j++) {
      data_buffer[i][j] = data_buffer[i + 1][j];
    }
  }
  buffer_index = WINDOW_SIZE - 1;
  
  return collision;
}

// ============================================
//              Main Loop
// ============================================
void loop() {
  // ============================================
  // استقبال بيانات من Serial
  // التنسيق: acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z
  // ============================================
  
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    
    // تجاهل الأسطر الفارغة أو التعليقات
    if (line.length() == 0 || line.startsWith("#") || line.startsWith("//")) {
      return;
    }
    
    // Parse البيانات
    int values_count = 0;
    float values[9];
    int start_pos = 0;
    
    for (int i = 0; i <= line.length(); i++) {
      if (i == line.length() || line[i] == ',' || line[i] == ' ' || line[i] == '\t') {
        if (i > start_pos && values_count < 9) {
          values[values_count++] = line.substring(start_pos, i).toFloat();
        }
        start_pos = i + 1;
      }
    }
    
    // التحقق من أن لدينا 9 قيم
    if (values_count == 9) {
      // معالجة البيانات
      preprocess_and_add_data(
        values[0], values[1], values[2],  // acc_x, acc_y, acc_z
        values[3], values[4], values[5],  // gyro_x, gyro_y, gyro_z
        values[6], values[7], values[8]   // mag_x, mag_y, mag_z
      );
      
      // تشغيل الـ inference
      run_inference();
    } else {
      Serial.print("⚠️  Invalid data format. Expected 9 values, got ");
      Serial.println(values_count);
    }
  }
  
  // ============================================
  // قراءة من IMU sensor مباشرة (أضف كودك هنا)
  // ============================================
  
  /* مثال لـ MPU6050:
  
  #include <MPU6050.h>
  MPU6050 mpu;
  
  void loop() {
    if (mpu.update()) {
      float acc_x = mpu.getAccX();
      float acc_y = mpu.getAccY();
      float acc_z = mpu.getAccZ();
      float gyro_x = mpu.getGyroX();
      float gyro_y = mpu.getGyroY();
      float gyro_z = mpu.getGyroZ();
      float mag_x = mpu.getMagX();
      float mag_y = mpu.getMagY();
      float mag_z = mpu.getMagZ();
      
      preprocess_and_add_data(acc_x, acc_y, acc_z, 
                              gyro_x, gyro_y, gyro_z,
                              mag_x, mag_y, mag_z);
      run_inference();
    }
    delay(SAMPLING_RATE_MS);
  }
  */
  
  // ============================================
  // أوامر Serial للتحكم
  // ============================================
  
  // يمكنك إضافة أوامر مثل:
  // "STATS" - لعرض الإحصائيات
  // "RESET" - لإعادة ضبط الإحصائيات
  // "TEST" - لاختبار النظام
  
  delay(SAMPLING_RATE_MS);
}

// ============================================
//          Test Function
// ============================================
void test_with_random_data() {
  Serial.println("\n=================================");
  Serial.println("Testing with random data...");
  Serial.println("=================================\n");
  
  for (int i = 0; i < 20; i++) {
    // توليد بيانات عشوائية واقعية
    float acc_x = random(-150, 150) / 10.0;   // -15g to +15g
    float acc_y = random(-150, 150) / 10.0;
    float acc_z = random(-150, 150) / 10.0;
    float gyro_x = random(-2000, 2000) / 10.0; // -200 to +200 deg/s
    float gyro_y = random(-2000, 2000) / 10.0;
    float gyro_z = random(-2000, 2000) / 10.0;
    float mag_x = random(-500, 500) / 10.0;    // -50 to +50 μT
    float mag_y = random(-500, 500) / 10.0;
    float mag_z = random(-500, 500) / 10.0;
    
    Serial.print("Sample ");
    Serial.print(i + 1);
    Serial.print("/20: ");
    Serial.print(acc_x, 1);
    Serial.print(", ");
    Serial.print(acc_y, 1);
    Serial.print(", ");
    Serial.println(acc_z, 1);
    
    preprocess_and_add_data(acc_x, acc_y, acc_z, 
                           gyro_x, gyro_y, gyro_z, 
                           mag_x, mag_y, mag_z);
    
    if (i >= WINDOW_SIZE - 1) {
      run_inference();
    }
    
    delay(SAMPLING_RATE_MS);
  }
  
  Serial.println("\n=================================");
  Serial.println("Test completed!");
  Serial.println("=================================\n");
}

// ============================================
//          Helper Functions
// ============================================

void print_statistics() {
  Serial.println("\n=================================");
  Serial.println("System Statistics");
  Serial.println("=================================");
  Serial.print("Total inferences: ");
  Serial.println(total_inferences);
  Serial.print("Total collisions: ");
  Serial.println(total_collisions);
  Serial.print("Collision rate: ");
  if (total_inferences > 0) {
    Serial.print((total_collisions * 100.0) / total_inferences, 2);
    Serial.println("%");
  } else {
    Serial.println("N/A");
  }
  Serial.print("Average inference time: ");
  Serial.print(avg_inference_time, 2);
  Serial.println(" ms");
  Serial.print("Free heap: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");
  Serial.println("=================================\n");
}

void reset_statistics() {
  total_inferences = 0;
  total_collisions = 0;
  avg_inference_time = 0;
  Serial.println("✓ Statistics reset");
}
