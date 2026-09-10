import argparse, json, sys
from pathlib import Path
from engine.video import VideoAnalyzer

def main():
    p=argparse.ArgumentParser(description='Run the GEG Field Intelligence analyzer on one video.')
    p.add_argument('video')
    p.add_argument('--stride',type=int,default=3)
    p.add_argument('--imgsz',type=int,default=512)
    args=p.parse_args()
    incidents,meta=VideoAnalyzer(detection_stride=args.stride,imgsz=args.imgsz).analyze(args.video)
    print(json.dumps({'meta':meta,'incident_count':len(incidents),'behaviours':sorted(set(x['behaviour'] for x in incidents))},indent=2))

if __name__=='__main__': main()
