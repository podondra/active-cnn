# Active CNN

Code for Astronomy &amp; Astrophysics letter

	$ docker build -t podondra/active-cnn .
	$ docker run --runtime=nvidia \
		-it \
		-v /data/podondra/active-cnn:/notebooks \
		-v /data/public/LAMOST-DR2:/lamost:ro \
		-p 8888:8888 \
		--name podondra-active-cnn \
		podondra/active-cnn
