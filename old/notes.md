to process output files for manual checking, on local machine:

```
cd /Users/owenfriend/Documents/temple_local/analysis/behavior
organize_movie_data.sh txt
organize_movie_data.sh wav
```

to transcribe audios in parallel on tacc:
* will use gpu if available, cpu on development/compute nodes is faster if only doing a subset of subjects because of longer gpu queues
```
transcribe_gpu.sh $SUB
```
