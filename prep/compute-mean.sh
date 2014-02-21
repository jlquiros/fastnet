#!/bin/bash

export PATH=$PATH:/home/power/w/hadoop-1.1.0/bin/

hadoop fs -rmr /imagenet/metadata/image-mean-256

hadoop jar /home/power/w/hadoop-1.1.0/build/contrib/streaming/hadoop-streaming-1.1.0-SNAPSHOT.jar\
 -input /imagenet/batches/imagesize-256\
 -output /imagenet/metadata/image-mean-256 \
 -mapper "/home/power/w/fastnet/prep/imagenet_mr.py compute_mean_mapper"\
 -reducer "/home/power/w/fastnet/prep/imagenet_mr.py compute_mean_reducer"\
 -jobconf mapred.reduce.tasks=1\
 -inputformat org.rjpower.hadoop.ZipFileInputFormat\
 -outputformat org.rjpower.hadoop.ZipFileOutputFormat\
 -io rawbytes 
