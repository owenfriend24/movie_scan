to process output files for manual checking, on local machine:

```
cd /Users/owenfriend/Documents/temple_local/analysis/behavior
organize_movie_data.sh txt
organize_movie_data.sh wav
```

to transcribe audios in parallel on tacc:
* will use gpu if available, I think cpu on development nodes is faster in the long run though because of longer gpu queues
```
transcribe_gpu.sh $SUB
```
