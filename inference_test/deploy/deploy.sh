# minikube start --force --cpus=1
eval $(minikube -p minikube docker-env)
docker build -t albert-inference-job-minikube:latest ../source
kubectl apply -f minikube.yaml
kubectl port-forward svc/albert-inference-service 5000:5000