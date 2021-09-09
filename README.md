# Finger-Counter
I used convexHull in OpenCV to count the numbers being held up

This progam detects a hand in your webcam and then thresholds the frame to make it binary with your hand being white and the background being black. The program then
draws a circle that is centered at the center of your hand, with a radius that is just short of the edge of your hand. This circle is also white. Performing a bitwise AND with
the circle and your hand contour creates multiple contours that represent where your hand intersects with the circle. These sections should in theory be your fingers. We count
the contours that are not your wrist to count fingers.

Limitations:
This program is limited by the lighting in your room. Since we threshold the frame before grabbing contours, the brightness of your hand can interfere with the contours 
being detected leading to inconsistencies with the finger counting. This also makes it difficult to detect zero fingers.

Since the thumb is shorter than the rest of your fingers, and the circle is drawn based on the most extreme points of your hand, which is most likely your middle or ring finger,
the thumb sometimes has a difficult time intersecting with the circle. Turning your hand in different angles helps with this. 
