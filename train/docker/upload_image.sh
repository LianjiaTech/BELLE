export https_proxy=...
export http_proxy=...
export all_proxy=...
docker_user=...

docker tag transformers:ds $docker_user/transformers:ds_$(date +%Y%m%d)
docker push $docker_user/transformers:ds_$(date +%Y%m%d)
docker tag belle:$(date +%Y%m%d) $docker_user/belle:$(date +%Y%m%d)
docker push $docker_user/belle:$(date +%Y%m%d)
docker tag belle:$(date +%Y%m%d) $docker_user/belle:latest
docker push $docker_user/belle:latest