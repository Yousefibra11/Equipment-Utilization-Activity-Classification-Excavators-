



# Mining Equipment Activity Classifier

## 1. Project Overview
This system uses Computer Vision to track Excavator and Truck utilization in real-time. It classifies activities and streams data to a PostgreSQL database.

## 2. The Articulated Equipment Challenge (How I solved it)
Standard bounding boxes fail when an excavator is stationary but its arm is moving. 
- **The Solution:** I implemented Instance Segmentation to track the **Bucket** as a separate entity.
- **Arm-Only Motion:** By calculating the Euclidean distance of the bucket centroid across 15-frame intervals, the system detects movement even if the main chassis (the body of the excavator) is still.
- **Activity Logic:** - **Digging:** Bucket is below the horizontal mid-line of the excavator.
    - **Dumping:** Bucket centroid enters the "Tight Zone" (100px padding) of a detected truck.

## 3. Data Pipeline
- **Vision:** YOLOv8 (Custom Segment Model + COCO).
- **Dashboard:** Streamlit UI.
- **Storage:** PostgreSQL (Default 'postgres' DB).

Mining Equipment Activity Classifier
 System ArchitectureThis project is built as a modular microservice pipeline designed for real-time edge deployment.Vision Engine: YOLOv8-segmentation (Custom trained for Mining Equipment).Message Broker: Apache Kafka (Handles high-throughput telemetry streams).Storage: PostgreSQL (Time-series data for equipment utilization).Frontend: Streamlit (Real-time monitoring dashboard).
🧠 Technical Deep-Dive: The Articulated Equipment Challenge1. The "Stationary-but-Active" ProblemIn heavy industry, an excavator is often "Active" while its tracks are stationary. Standard object detection (bounding boxes) fails here because the box coordinates $(x, y)$ of the machine don't change, leading to false "Idle" reports.2. The Solution: Centroid-Based Segment TrackingI implemented Instance Segmentation to treat the Excavator as an articulated assembly rather than a single block.Mask Isolation: The model isolates the Bucket mask as a sub-entity.Euclidean Motion Detection: Even if the chassis movement is $0$, the system calculates the movement of the bucket centroid across a 15-frame buffer using the distance formula:$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$Thresholding: If $d > 5$ pixels, the system overrides the "Stationary" status and marks the equipment as ACTIVE (Arm Motion).
Activity Classification LogicThe system uses Spatial Interaction Zones to classify behavior without requiring a separate "Action" model, making it faster and more efficient.ActivitySpatial TriggerMotion TriggerDIGGINGBucket is in the lower 40% of the frame.Bucket Centroid $d >$ threshold.DUMPINGBucket Centroid enters a 100px proximity zone of a Truck.Centroid deceleration detected.IDLEBucket and Chassis are within center-point noise limits.No movement detected for $> 5$ seconds.
Design Decisions & Trade-offsDecisionBenefit (The "Why")Trade-off (The "Cost")YOLOv8-SegHigh-precision tracking of articulated parts (buckets/arms).Higher computational cost (approx. 30% slower than YOLOv8-detect).Docker-ComposeGuaranteed environment parity for the grader/user.Requires significant RAM (approx. 4GB) to run the full stack.Kafka BrokerAllows the system to scale to 100+ cameras in a real mine.Adds networking complexity for local development.15-Frame BufferEliminates "flicker" and false state changes.Introduces a 0.5s lag in the real-time UI.
