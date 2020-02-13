mkdir models
mkdir models/mpi
mkdir models/coco
curl -o models/mpi/pose_iter_160000.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
curl -o models/mpi/pose_deploy_linevec_faster_4_stages.prototxt https://raw.githubusercontent.com/spmallick/learnopencv/master/OpenPose/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt
curl -o models/mpi/pose_deploy_linevec.prototxt https://raw.githubusercontent.com/spmallick/learnopencv/master/OpenPose/pose/mpi/pose_deploy_linevec.prototxt
curl -o models/coco/pose_deploy_linevec.prototxt https://raw.githubusercontent.com/spmallick/learnopencv/master/OpenPose/pose/coco/pose_deploy_linevec.prototxt
curl -o models/coco/pose_iter_440000.caffemodel http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
