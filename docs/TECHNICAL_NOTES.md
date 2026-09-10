# Technical notes

## Perception
The upload path uses Ultralytics YOLOWorld with open-vocabulary prompts covering people, cartons/packages, common warehouse products and handling equipment. This avoids filename-based assumptions and avoids a closed COCO-only object vocabulary.

## Tracking
Detections are associated across frames using persistent track IDs, IoU and centroid distance. Every track retains temporal history: position, velocity, speed, box geometry and confidence.

## Behaviour reasoning
Behaviour events are produced only from observed tracked entities and temporal/spatial evidence. The detectors do not inspect the video filename and do not assign behaviours to timestamps because a challenge video is known.

## Risk semantics
`Observed behaviour -> potential risk -> human intervention`. `damage_confirmed` remains false unless an independently validated damage detector is added.

## Speed
The default detector stride is 3. This means perception runs on roughly every third frame while the tracker preserves object identity between perception updates. Increase stride for faster CPU analysis; reduce it for higher temporal resolution.

## Known prototype limits
Open-vocabulary detection is not a substitute for a warehouse-specific labelled training set. Some behaviours, especially strap-as-handle, wet-floor condition and true product weight, cannot be reliably inferred from generic RGB video alone without additional sensors or trained models. The system therefore uses conservative visual proxies and does not claim unavailable physical facts.
