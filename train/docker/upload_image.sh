export https_proxy=...
export http_proxy=...
export all_proxy=...
docker_user=...
tag=$(date +%Y%m%d)

docker tag transformers:ds $docker_user/transformers:ds_$tag
docker push $docker_user/transformers:ds_$tag
docker tag transformers:ds $docker_user/transformers:latest
docker push $docker_user/transformers:latest

docker tag belle $docker_user/belle:$tag
docker push $docker_user/belle:$tag
docker tag belle $docker_user/belle:latest
docker push $docker_user/belle:latest

docker tag roce $docker_user/roce:$tag
docker push $docker_user/roce:$tag
docker tag roce $docker_user/roce:latest
docker push $docker_user/roce:latest