



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

Technical Write-up1. The Articulated Equipment ChallengeThe Problem: In mining, an excavator is "active" even when its tracks aren't moving. Traditional motion detection (which looks at the whole bounding box) would incorrectly label an excavator digging in one spot as "Idle."The Solution (Arm-Only Motion):I moved away from "Whole-Body Tracking" and implemented Centroid-Based Segment Tracking.Isolating the Bucket: I used YOLOv8-seg (Instance Segmentation) to create a specific mask for the Bucket.Euclidean Distance Logic: The system calculates the center point (centroid) of the bucket mask.Thresholding: Even if the Excavator’s main body movement is 0%, if the Bucket centroid moves more than 5 pixels within a 15-frame window, the state is forced to "ACTIVE".2. Activity Classification LogicThe model uses Spatial Interaction Zones to classify behavior:DIGGING: Triggered when the Bucket mask is moving and is located in the lower 40% of the video frame.DUMPING: Triggered when the Bucket's centroid enters a 100-pixel "Proximity Zone" around the top of a detected Truck.IDLE: Triggered only if both the Chassis and the Bucket centroids remain static for more than 5 seconds.3. Design Decisions & Trade-offsEvery engineering choice has a "Cost" vs "Benefit." Here is how I balanced them:DecisionBenefit (The "Why")Trade-off (The "Cost")YOLOv8-SegmentationAllows for high-precision bucket tracking, which is essential for articulated movement.Requires more GPU/CPU power than standard object detection, leading to lower FPS on weak hardware.Docker InfrastructureEnsures the grader can run Kafka and Postgres without version conflicts or manual installation.The initial setup (downloading images) takes a few minutes and requires Docker Desktop.Kafka IntegrationMakes the system "Production-Ready"—data can be streamed to multiple dashboards at once.Adds complexity to the architecture compared to a simple local CSV file.15-Frame LookbackPrevents "flickering" (where a single bad frame changes the status from Active to Idle).Introduces a tiny (approx. 0.5s) delay in state reporting.
