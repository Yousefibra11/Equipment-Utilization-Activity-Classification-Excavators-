



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

Mining Equipment Activity ClassifierReal-time Edge Intelligence for Production Tracking
System ArchitectureThis project is built as a modular microservice pipeline designed for real-time edge deployment. It separates heavy vision processing from data storage to ensure system stability.Vision Engine: YOLOv8-segmentation (Custom trained for Mining Equipment).Message Broker: Apache Kafka (Handles high-throughput telemetry streams).Storage: PostgreSQL (Time-series data for equipment utilization).Frontend: Streamlit (Real-time monitoring dashboard).

🧠 The Articulated Equipment ChallengeThe Problem: In heavy industry, an excavator is often "Active" while its tracks are stationary. Standard object detection (bounding boxes) fails here because the box coordinates $(x,y)$ of the machine don't change, leading to false "Idle" reports.

🛡️ The Technical Solution: Centroid-Based Segment TrackingI implemented Instance Segmentation to treat the Excavator as an articulated assembly rather than a single static block.Mask Isolation: The model isolates the Bucket mask as a unique sub-entity.Euclidean Motion Detection: Even if the chassis movement is $0$, the system calculates the movement of the bucket centroid across a 15-frame buffer using the distance formula:$$
d = \sqrt{(x_t - x_{t-15})^2 + (y_t - y_{t-15})^2}
$$ 
Dynamic Thresholding: If $d > 5$ pixels, the system overrides the "Stationary" status and marks the equipment as ACTIVE (Arm Motion).🚦 Activity Classification LogicInstead of a single "Action" model, the system uses Spatial Interaction Zones for high-speed inference.⛏️ DIGGINGZone: Bucket is in the lower 40% of the frame.Trigger: Bucket Centroid movement d >threshold.

DUMPINGZone: Bucket Centroid enters a 100px proximity zone above a Truck.Trigger: Centroid deceleration and spatial overlap detected.
IDLEZone: Bucket and Chassis remain within center-point noise limits.Trigger: No movement detected for $> 5$ seconds.
Design Decisions & Trade-offsYOLOv8-Segmentation vs. Standard DetectionBenefit: 
Allows for high-precision tracking of articulated parts (buckets/arms).
Trade-off: Higher computational cost (approx. 30% slower than YOLOv8-detect).
Docker-Compose vs. Manual SetupBenefit: Guaranteed environment parity; 
Trade-off: Requires significant local RAM (approx. 4GB) to run the Kafka and Postgres containers.
Apache Kafka vs. Local CSVBenefit: Makes the system "Production-Ready"—data can be streamed to multiple dashboards simultaneously;
Trade-off: Adds networking complexity for local development.
15-Frame Temporal BufferBenefit: Eliminates "flicker" and false state changes caused by lighting or minor occlusions;
Trade-off: Introduces a minor 0.5s lag in the real-time UI.
