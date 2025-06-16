/*********************************************************************
 *  4WD Robot Control System - ROS2 Interface
 *  
 *  Features:
 *  - Distance-based movement control with PD feedback
 *  - IMU orientation sensing (MPU6050)
 *  - Environmental monitoring (BME280)
 *  - Current sensing (ACS758)
 *  - Bumper collision detection
 *  - JSON sensor data output for ROS2
 *  
 *  Commands:
 *  - MOVE,<distance_meters>  : Move forward/backward by distance
 *  - TURN,<angle_degrees>    : Turn by angle using IMU
 *  - PWM,<left>,<right>      : Manual PWM control (-255 to 255)
 *  - TURN_LEFT / TURN_RIGHT  : Fixed-speed turning
 *  - STOP                    : Emergency stop
 *  - STATUS                  : Print current status
 *  
 *  Hardware:
 *  - Arduino Mega 2560
 *  - 4x DC motors with Adafruit motor drivers
 *  - 2x encoders on front wheels
 *  - MPU6050 IMU sensor
 *  - BME280 environment sensor
 *  - ACS758 current sensor
 *  - 4x bumper switches
 *  - 2x LDR light sensors
 *********************************************************************/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

/* ---------------- Pin assignments ---------------- */
#define RELAY_PIN        48
#define TOP_BUMPER_PIN   44 
#define BOTTOM_BUMPER_PIN 42
#define LEFT_BUMPER_PIN  40
#define RIGHT_BUMPER_PIN 38
#define CURRENT_OUT_PIN  A0
#define CURRENT_IN_PIN   A1
#define LDR_LEFT_PIN     A3
#define LDR_RIGHT_PIN    A2

// Motor control pins
#define L_MOTOR_DIR 6   
#define L_MOTOR_PWM 7
#define L_MOTOR2_DIR 8  
#define L_MOTOR2_PWM 9
#define R_MOTOR_DIR 10   
#define R_MOTOR_PWM 11
#define R_MOTOR2_DIR 12 
#define R_MOTOR2_PWM 13

// Encoder pins (front motors only)
const uint8_t ENC_LEFT_A = 3, ENC_LEFT_B = 5;
const uint8_t ENC_RIGHT_A = 2, ENC_RIGHT_B = 4;

/* ---------------- Control Variables ---------------- */
// PD control for distance movement
const float Kp = 0.025, Kd = 0.001;
const int MAX_SPEED = 255, MIN_SPEED = 70;
const float LEFT_SCALE = 1.0, RIGHT_SCALE = 2.5;

// Control states
enum ControlMode { IDLE, MOVING, TURNING, MANUAL };
ControlMode currentMode = IDLE;
bool isMoving = false;  // Motion indicator for serial output

// Distance control
long targetDistance = 0;
long startEncoderLeft = 0, startEncoderRight = 0;
float prevErrorLeft = 0, prevErrorRight = 0;

// Turn control
const int TURN_SPEED = 130;
float targetYaw = 0;
int turnDirection = 0;  // 1=right, -1=left

// Manual control
int manualLeftPWM = 0, manualRightPWM = 0;

// Timing
unsigned long moveStartTime = 0;
const unsigned long MOVE_TIMEOUT = 25000;

// Encoder variables
volatile long encoderLeft = 0, encoderRight = 0;
volatile int lastStateLeft = LOW, lastStateRight = LOW;

// Control timing
unsigned long lastControlUpdate = 0;
const unsigned long CONTROL_INTERVAL = 50; // Control every 50ms

// Sensor variables
Adafruit_MPU6050 mpu;
Adafruit_BME280 bme;
float roll = 0, pitch = 0, yaw = 0;
bool imuWorking = false, bmeWorking = false;
float ACS_OFFSET_V = 2.525;
const float ACS_SENSITIVITY = 0.040;

// Timing
unsigned long lastIMUUpdate = 0, lastSensorUpdate = 0;
const unsigned long IMU_PERIOD_MS = 50, SENSOR_PERIOD_MS = 100;

// Bumper collision handling
bool inCollisionRecovery = false;

/* ========================================================================= */
/*                                SETUP                                      */
/* ========================================================================= */
void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  
  pinMode(TOP_BUMPER_PIN, INPUT_PULLUP);
  pinMode(BOTTOM_BUMPER_PIN, INPUT_PULLUP);
  pinMode(LEFT_BUMPER_PIN, INPUT_PULLUP);
  pinMode(RIGHT_BUMPER_PIN, INPUT_PULLUP);

  // Motor pins
  pinMode(L_MOTOR_DIR, OUTPUT); pinMode(L_MOTOR_PWM, OUTPUT);
  pinMode(L_MOTOR2_DIR, OUTPUT); pinMode(L_MOTOR2_PWM, OUTPUT);
  pinMode(R_MOTOR_DIR, OUTPUT); pinMode(R_MOTOR_PWM, OUTPUT);
  pinMode(R_MOTOR2_DIR, OUTPUT); pinMode(R_MOTOR2_PWM, OUTPUT);

  // Encoder pins
  pinMode(ENC_LEFT_A, INPUT_PULLUP); pinMode(ENC_LEFT_B, INPUT_PULLUP);
  pinMode(ENC_RIGHT_A, INPUT_PULLUP); pinMode(ENC_RIGHT_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_LEFT_A), isrEncLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RIGHT_A), isrEncRight, CHANGE);

  // Initialize sensors
  imuWorking = (mpu.begin(0x68) || mpu.begin(0x69));
  if (imuWorking) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  }
  
  bmeWorking = (bme.begin(0x76) || bme.begin(0x77));
  calibrateCurrentSensor();

  Serial.println("READY");
}

/* ========================================================================= */
/*                                MAIN LOOP                                  */
/* ========================================================================= */
void loop() {
  // Handle serial commands
  if (Serial.available()) handleSerialCommand();

  // Check for bumper collisions (automatic backup)
  if (!inCollisionRecovery && (digitalRead(TOP_BUMPER_PIN) == LOW || 
                               digitalRead(BOTTOM_BUMPER_PIN) == LOW || 
                               digitalRead(LEFT_BUMPER_PIN) == LOW || 
                               digitalRead(RIGHT_BUMPER_PIN) == LOW)) {
    inCollisionRecovery = true;
    startBackup();
  }

  // Update IMU
  if (imuWorking && millis() - lastIMUUpdate >= IMU_PERIOD_MS) {
    lastIMUUpdate = millis();
    updateIMU();
  }

  // Auto-reset yaw when idle
  static bool wasIdle = false;
  bool isIdle = (currentMode == IDLE);
  if (isIdle && !wasIdle) yaw = 0;
  wasIdle = isIdle;

  // Update motion state
  isMoving = (currentMode == MOVING || currentMode == TURNING || 
              (currentMode == MANUAL && (manualLeftPWM != 0 || manualRightPWM != 0)));

  // Run control
  if (millis() - lastControlUpdate >= CONTROL_INTERVAL) {
    lastControlUpdate = millis();
    
    switch (currentMode) {
      case MOVING: runDistanceControl(); break;
      case TURNING: runTurnControl(); break;
      case MANUAL: runManualControl(); break;
      default: stopMotors(); break;
    }
  }

  // Send sensor data
  if (millis() - lastSensorUpdate >= SENSOR_PERIOD_MS) {
    lastSensorUpdate = millis();
    sendSensorData();
  }
}

/* ========================================================================= */
/*                          COMMAND HANDLING                                 */
/* ========================================================================= */
void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();  // Make command case-insensitive
  
  if (command == "STOP") {
    emergencyStop();
    return;
  }
  
  if (command.startsWith("MOVE,")) {
    float distanceMeters = command.substring(5).toFloat();
    const float WHEEL_CIRCUMFERENCE_M = 3.14159 * 0.192;
    const float ENCODER_COUNTS_PER_REV = 9000.0;
    long distanceEncoderCounts = (long)((distanceMeters / WHEEL_CIRCUMFERENCE_M) * ENCODER_COUNTS_PER_REV);
    
    encoderLeft = 0;
    encoderRight = 0;
    startMove(distanceEncoderCounts);
  }
  else if (command.startsWith("TURN,")) {
    float targetAngle = command.substring(5).toFloat();
    yaw = 0;
    startTurn(targetAngle);
  }
  else if (command.startsWith("PWM,")) {
    // PWM,left,right - Manual PWM control for joystick
    int comma1 = command.indexOf(',');        // First comma (after PWM)
    int comma2 = command.indexOf(',', comma1 + 1);  // Second comma
    
    if (comma1 > 0 && comma2 > 0) {
      manualLeftPWM = command.substring(comma1 + 1, comma2).toInt();
      manualRightPWM = command.substring(comma2 + 1).toInt();
      currentMode = MANUAL;
    }
  }
  else if (command == "TURN_LEFT") {
    // Fixed turn left for joystick
    currentMode = MANUAL;
    manualLeftPWM = TURN_SPEED;     // Left forward
    manualRightPWM = -TURN_SPEED;   // Right backward
  }
  else if (command == "TURN_RIGHT") {
    // Fixed turn right for joystick  
    currentMode = MANUAL;
    manualLeftPWM = -TURN_SPEED;    // Left backward
    manualRightPWM = TURN_SPEED;    // Right forward
  }
  else if (command == "STATUS") {
    printStatus();
  }
}

void startMove(long distance) {
  targetDistance = distance;
  startEncoderLeft = encoderLeft;
  startEncoderRight = encoderRight;
  prevErrorLeft = 0;
  prevErrorRight = 0;
  currentMode = MOVING;
  moveStartTime = millis();
}

void startTurn(float angle) {
  yaw = 0;
  targetYaw = angle;
  turnDirection = (angle > 0) ? 1 : -1;
  currentMode = TURNING;
  moveStartTime = millis();
}

void startBackup() {
  // Automatic 1-meter backup when bumper is hit
  const float WHEEL_CIRCUMFERENCE_M = 3.14159 * 0.192;
  const float ENCODER_COUNTS_PER_REV = 9000.0;
  long backupDistance = (long)((-1.0 / WHEEL_CIRCUMFERENCE_M) * ENCODER_COUNTS_PER_REV);
  
  encoderLeft = 0;
  encoderRight = 0;
  startMove(backupDistance);

  
}

void emergencyStop() {
  currentMode = IDLE;
  targetDistance = 0;
  targetYaw = 0;
  inCollisionRecovery = false;
  stopMotors();
}

/* ========================================================================= */
/*                           DISTANCE CONTROL                                */
/* ========================================================================= */
void runDistanceControl() {
  if (currentMode != MOVING) {
    stopMotors();
    return;
  }

  // Safety timeout
  if (millis() - moveStartTime > MOVE_TIMEOUT) {
    stopMotors();
    currentMode = IDLE;
    inCollisionRecovery = false;
    return;
  }

  // Calculate individual wheel distances
  long leftDistance = encoderLeft - startEncoderLeft;
  long rightDistance = encoderRight - startEncoderRight;

  // Individual wheel errors
  float errorLeft = targetDistance - leftDistance;
  float errorRight = targetDistance - rightDistance;
  
  // Large tolerance for high-resolution encoders (1000 counts ~= 10cm)
  const int TOLERANCE = 1000;
  
  // Check if either wheel reached target - stop both wheels
  if (abs(errorLeft) < TOLERANCE || abs(errorRight) < TOLERANCE) {
    stopMotors();
    currentMode = IDLE;
    inCollisionRecovery = false;
    return;
  }

  // Calculate derivatives
  float derivativeLeft = errorLeft - prevErrorLeft;
  float derivativeRight = errorRight - prevErrorRight;

  // Reduce gains when close to target to prevent oscillation
  float kp = Kp;
  float kd = Kd;
  float avgError = (abs(errorLeft) + abs(errorRight)) / 2;
  if (avgError < 2000) {
    kp *= 0.5;  // Reduce proportional gain when close
    kd *= 0.5;  // Reduce derivative gain when close
  }

  // PD control for each wheel with reduced gains when close
  float outputLeft = kp * errorLeft + kd * derivativeLeft;
  float outputRight = kp * errorRight + kd * derivativeRight;

  // Pitch-based speed adjustment
  int maxSpeed = MAX_SPEED;
  if (pitch < -8.0) {
    // Going uphill - use full speed
    maxSpeed = MAX_SPEED;  // Use full speed
  } else if (pitch > 8.0) {
    // Going downhill - reduce speed significantly
    maxSpeed = 150;  // Cap at 150 PWM
  }

  // Speed limiting when close to target
  float distanceToTarget = avgError;
  int speedLimit = maxSpeed;
  if (distanceToTarget < 3000) {
    speedLimit = map(distanceToTarget, 0, 3000, MIN_SPEED, maxSpeed/2);
  }

  // Convert to PWM with scaling
  int pwmLeft = (int)constrain(abs(outputLeft * LEFT_SCALE), 0, speedLimit);
  int pwmRight = (int)constrain(abs(outputRight * RIGHT_SCALE), 0, speedLimit);
  
  // Apply minimum speed if moving
  if (pwmLeft > 0 && pwmLeft < MIN_SPEED) pwmLeft = MIN_SPEED;
  if (pwmRight > 0 && pwmRight < MIN_SPEED) pwmRight = MIN_SPEED;
  
  // Apply direction
  if (errorLeft < 0) pwmLeft = -pwmLeft;
  if (errorRight < 0) pwmRight = -pwmRight;

  // Apply to motors
  setMotors(pwmLeft, pwmRight);

  // Update previous errors
  prevErrorLeft = errorLeft;
  prevErrorRight = errorRight;
}

/* ========================================================================= */
/*                             TURN CONTROL                                  */
/* ========================================================================= */
void runTurnControl() {
  if (currentMode != TURNING) {
    stopMotors();
    return;
  }

  // Safety timeout
  if (millis() - moveStartTime > MOVE_TIMEOUT) {
    stopMotors();
    currentMode = IDLE;
    return;
  }

  // Check if reached target
  float yawError = targetYaw - yaw;
  if (abs(yawError) < 2.0) {
    stopMotors();
    currentMode = IDLE;
    return;
  }

  // Apply turn motors
  if (turnDirection == 1) {
    setMotorsForTurn(TURN_SPEED, -TURN_SPEED);  // Right turn
  } else {
    setMotorsForTurn(-TURN_SPEED, TURN_SPEED);  // Left turn
  }
}

void runManualControl() {
  // If both PWM are 0, go back to idle
  if (manualLeftPWM == 0 && manualRightPWM == 0) {
    currentMode = IDLE;
    stopMotors();
    return;
  }
  
  setMotorsForTurn(manualLeftPWM, manualRightPWM);
}

/* ========================================================================= */
/*                           MOTOR CONTROL                                   */
/* ========================================================================= */
void setMotors(int leftSpeed, int rightSpeed) {
  // Determine overall direction based on average speed
  int avgSpeed = (leftSpeed + rightSpeed) / 2;
  
  // Set ALL motors to same direction based on average
  if (avgSpeed >= 0) {
    // ALL MOTORS FORWARD
    digitalWrite(L_MOTOR_DIR, LOW);   // Left front forward
    digitalWrite(L_MOTOR2_DIR, LOW);  // Left back forward
    digitalWrite(R_MOTOR_DIR, HIGH);  // Right front forward (inverted)
    digitalWrite(R_MOTOR2_DIR, HIGH); // Right back forward (inverted)
    
    // Apply PWM - invert right side PWM because direction is HIGH
    analogWrite(L_MOTOR_PWM, abs(leftSpeed));      // Normal PWM for left (direction LOW)
    analogWrite(L_MOTOR2_PWM, abs(leftSpeed));
    analogWrite(R_MOTOR_PWM, 255 - abs(rightSpeed)); // Inverted PWM for right (direction HIGH)
    analogWrite(R_MOTOR2_PWM, 255 - abs(rightSpeed));
  } else {
    // ALL MOTORS BACKWARD
    digitalWrite(L_MOTOR_DIR, HIGH);  // Left front backward
    digitalWrite(L_MOTOR2_DIR, HIGH); // Left back backward
    digitalWrite(R_MOTOR_DIR, LOW);   // Right front backward (inverted)
    digitalWrite(R_MOTOR2_DIR, LOW);  // Right back backward (inverted)
    
    // Apply PWM - invert left side PWM because direction is HIGH
    analogWrite(L_MOTOR_PWM, 255 - abs(leftSpeed));  // Inverted PWM for left (direction HIGH)
    analogWrite(L_MOTOR2_PWM, 255 - abs(leftSpeed));
    analogWrite(R_MOTOR_PWM, abs(rightSpeed));        // Normal PWM for right (direction LOW)
    analogWrite(R_MOTOR2_PWM, abs(rightSpeed));
  }
}

void setMotorsForTurn(int leftSpeed, int rightSpeed) {
  // Direct motor control for turning - no averaging of direction
  
  // Left motors
  if (leftSpeed >= 0) {
    digitalWrite(L_MOTOR_DIR, LOW);   // Forward
    digitalWrite(L_MOTOR2_DIR, LOW);  // Forward
    analogWrite(L_MOTOR_PWM, abs(leftSpeed));
    analogWrite(L_MOTOR2_PWM, abs(leftSpeed));
  } else {
    digitalWrite(L_MOTOR_DIR, HIGH);  // Backward
    digitalWrite(L_MOTOR2_DIR, HIGH); // Backward
    analogWrite(L_MOTOR_PWM, 255 - abs(leftSpeed));
    analogWrite(L_MOTOR2_PWM, 255 - abs(leftSpeed));
  }

  // Right motors (with PWM inversion for direction HIGH)
  if (rightSpeed >= 0) {
    digitalWrite(R_MOTOR_DIR, HIGH);  // Forward (inverted)
    digitalWrite(R_MOTOR2_DIR, HIGH); // Forward (inverted)
    analogWrite(R_MOTOR_PWM, 255 - abs(rightSpeed)); // Inverted PWM
    analogWrite(R_MOTOR2_PWM, 255 - abs(rightSpeed));
  } else {
    digitalWrite(R_MOTOR_DIR, LOW);   // Backward (inverted)
    digitalWrite(R_MOTOR2_DIR, LOW);  // Backward (inverted)
    analogWrite(R_MOTOR_PWM, abs(rightSpeed)); // Normal PWM when direction=LOW
    analogWrite(R_MOTOR2_PWM, abs(rightSpeed));
  }
}

void stopMotors() {
  // Set all direction pins to LOW (same direction)
  digitalWrite(L_MOTOR_DIR, LOW);
  digitalWrite(L_MOTOR2_DIR, LOW);
  digitalWrite(R_MOTOR_DIR, LOW);
  digitalWrite(R_MOTOR2_DIR, LOW);
  
  // Set all PWM to 0
  analogWrite(L_MOTOR_PWM, 0);
  analogWrite(L_MOTOR2_PWM, 0);
  analogWrite(R_MOTOR_PWM, 0);
  analogWrite(R_MOTOR2_PWM, 0);
  
  // Reset encoders whenever motors stop
  encoderLeft = 0;
  encoderRight = 0;
}

/* ========================================================================= */
/*                              SENSOR UPDATES                               */
/* ========================================================================= */
void updateIMU() {
  if (!imuWorking) return;
  
  sensors_event_t a, g, temp;
  if (!mpu.getEvent(&a, &g, &temp)) {
    imuWorking = false;
    roll = pitch = yaw = 0.0;
    return;
  }
  
  float accel_magnitude = sqrt(a.acceleration.x * a.acceleration.x + 
                              a.acceleration.y * a.acceleration.y + 
                              a.acceleration.z * a.acceleration.z);
  
  if (accel_magnitude < 5.0 || accel_magnitude > 15.0) return;
  
  roll = atan2(a.acceleration.y, a.acceleration.z) * 180.0 / PI;
  pitch = atan2(-a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + 
                            a.acceleration.z * a.acceleration.z)) * 180.0 / PI;
  
  static unsigned long lastTime = 0;
  unsigned long currentTime = millis();
  
  if (lastTime > 0) {
    float dt = (currentTime - lastTime) / 1000.0;
    if (dt > 0 && dt < 1.0) {
      yaw += g.gyro.z * dt * 180.0 / PI;
    }
  }
  lastTime = currentTime;
}

void sendSensorData() {
  // Reset yaw to 0 when robot is stopped
  if (!isMoving) {
    yaw = 0;
  }
  
  // JSON sensor data for ROS2
  Serial.print("{");
  Serial.print("\"imu\":{\"roll\":");Serial.print(roll, 2);
  Serial.print(",\"pitch\":");Serial.print(pitch, 2);
  Serial.print(",\"yaw\":");Serial.print(yaw, 2);
  Serial.print("},\"encoders\":{\"left\":");Serial.print(encoderLeft);
  Serial.print(",\"right\":");Serial.print(encoderRight);
  Serial.print("\"current\":{\"in\":");Serial.print(readCurrentIn(), 2);
  Serial.print(",\"out\":");Serial.print(readCurrentOut(), 2);
  Serial.print("},\"ldr\":{\"left\":");Serial.print(analogRead(LDR_LEFT_PIN));
  Serial.print(",\"right\":");Serial.print(analogRead(LDR_RIGHT_PIN));
    Serial.print("},\"environment\":{");
    if (bmeWorking) {
      Serial.print("\"temperature\":");Serial.print(bme.readTemperature(), 1);
      Serial.print(",\"humidity\":");Serial.print(bme.readHumidity(), 1);
      Serial.print(",\"pressure\":");Serial.print(bme.readPressure() / 100.0F, 1);
    } else {
      Serial.print("\"temperature\":0,\"humidity\":0,\"pressure\":0");
    }
    Serial.print("},\"bumpers\":{\"top\":");Serial.print(digitalRead(TOP_BUMPER_PIN) == LOW ? 1 : 0);
    Serial.print(",\"bottom\":");Serial.print(digitalRead(BOTTOM_BUMPER_PIN) == LOW ? 1 : 0);
    Serial.print(",\"left\":");Serial.print(digitalRead(LEFT_BUMPER_PIN) == LOW ? 1 : 0);
    Serial.print(",\"right\":");Serial.print(digitalRead(RIGHT_BUMPER_PIN) == LOW ? 1 : 0);
    Serial.print("},\"motion\":\"");
    Serial.print(isMoving ? "moving" : "stop");
  Serial.println("}");
}

/* ========================================================================= */
/*                           SENSOR HELPERS                                  */
/* ========================================================================= */
void calibrateCurrentSensor() {
  long sum = 0;
  for (int i = 0; i < 128; i++) {
    sum += analogRead(CURRENT_OUT_PIN);
    delayMicroseconds(120);
  }
  ACS_OFFSET_V = (sum / 128.0) * (5.0 / 1023.0);
}

float readCurrentOut() {
  float vout = analogRead(CURRENT_OUT_PIN) * (5.0 / 1023.0);
  return (vout - ACS_OFFSET_V) / ACS_SENSITIVITY;
}

float readCurrentIn() {
  float vout = analogRead(CURRENT_IN_PIN) * (5.0 / 1023.0);
  return (vout - ACS_OFFSET_V) / ACS_SENSITIVITY;
}

void printStatus() {
  Serial.print("STATUS:");
  Serial.print(currentMode);
  Serial.print(",");
  Serial.print(targetDistance);
  Serial.print(",");
  Serial.print(targetYaw);
  Serial.print(",");
  Serial.println(yaw);
}

/* ========================================================================= */
/*                           ENCODER INTERRUPTS                              */
/* ========================================================================= */
void isrEncLeft() {
  int a = digitalRead(ENC_LEFT_A);
  int b = digitalRead(ENC_LEFT_B);
  if (a != lastStateLeft) {
    if (a == b) encoderLeft++;
    else encoderLeft--;
    lastStateLeft = a;
  }
}

void isrEncRight() {
  int a = digitalRead(ENC_RIGHT_A);
  int b = digitalRead(ENC_RIGHT_B);
  if (a != lastStateRight) {
    if (a == b) encoderRight--;  // Inverted direction
    else encoderRight++;
    lastStateRight = a;
  }
}
