for FILE in models/skipgram/*
    do
	echo $FILE
	python evaluate.py $FILE
	echo "============="
    done

