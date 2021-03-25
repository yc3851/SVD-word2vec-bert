for FILE in models/svd/svd*
    do
	echo $FILE
	python evaluate.py $FILE
	echo "============="
    done

