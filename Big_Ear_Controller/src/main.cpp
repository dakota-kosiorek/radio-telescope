#include <Arduino.h>
#include <Stepper.h>
#include "Arduino_LED_Matrix.h"
#include "pwm.h"

ArduinoLEDMatrix matrix;
//PwmOut stepper(9);

void moveStepper(int desiredPos);

int currentPos = 0;


byte frame[8][12] = {
  { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
  { 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0 },
  { 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0 },
  { 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0 },
  { 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0 },
  { 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0 },
  { 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0 },
  { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 }
};

void setup() {
  Serial.begin(9600);
  matrix.begin();
  pinMode(9,OUTPUT);
  pinMode(10,OUTPUT);
  //stepper.period_us(10);
  digitalWrite(10, LOW);
  // stepper.begin((float)100,(float)50.0);
  // stepper.pulseWidth_us(3);
  
}

void loop(){
  moveStepper(0);
  delay(2000);
  moveStepper(90);
  delay(2000);
  moveStepper(180);
  delay(2000);
  moveStepper(270);
  delay(2000);
  moveStepper(360);
  delay(2000);
  moveStepper(270);
  delay(2000);
  moveStepper(180);
  delay(2000);
  moveStepper(90);
  delay(2000);

}

void moveStepper(int desiredPos){
  int numSteps;
  if (currentPos != desiredPos){
    numSteps = ceil(abs(desiredPos - currentPos)/1.8)*10;
    if (desiredPos > currentPos){
      digitalWrite(10, HIGH);
    }
    if ((desiredPos < currentPos)){
      digitalWrite(10,LOW);
    }

    for (int i = 0; i<numSteps; i++)
    {
      digitalWrite(9,HIGH);
      delayMicroseconds(3);
      digitalWrite(9,LOW);
      delayMicroseconds(51425);
      Serial.println(i);
    }
    currentPos = desiredPos;
  }

}
