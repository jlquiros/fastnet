#!/bin/bash

python build-synid-map.py

export PATH=$PATH:/home/power/w/hadoop-1.1.0/bin/
DST=/imagenet/batches/imagesize-256
hadoop fs -rmr $DST

hadoop jar /home/power/w/hadoop-1.1.0/build/contrib/streaming/hadoop-streaming-1.1.0-SNAPSHOT.jar\
 -input  '/imagenet/zip/*.zip'\
 -output $DST\
 -mapper "/home/power/w/fastnet/prep/imagenet_mr.py resize_and_crop_mapper"\
 -reducer org.apache.hadoop.mapred.lib.IdentityReducer\
 -jobconf mapred.reduce.tasks=650\
 -inputformat org.rjpower.hadoop.ZipFileInputFormat\
 -outputformat org.rjpower.hadoop.ZipFileOutputFormat\
 -io rawbytes 

hadoop fs -copyFromLocal ./batches.meta $DST
