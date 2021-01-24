pipeline {
     agent any
     stages {


       stage('Pre-clean-up of docker images') {
           steps {
                  sh 'df -h'
                  sh 'sudo docker system prune -f -a'
                  sh 'df -h'
            }
           }




         stage('Linting') {
             steps {
                    sh 'python3 -m venv venv'
                    sh '. venv/bin/activate'
                    sh 'sudo make install'

                    sh 'sudo wget -O /bin/hadolint https://github.com/hadolint/hadolint/releases/download/v1.16.3/hadolint-Linux-x86_64'
                    sh 'sudo chmod +x /bin/hadolint'
//                    sh 'make lint'
              }
             }




         stage('Build Docker Image') {
             steps {
                    sh 'sudo ./build_docker.sh; sleep 10'

              }
             }

         stage('Upload docker image to DockerHub') {
             steps {

                  withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'USER', passwordVariable: 'PASS')])
                {
                  sh 'echo $USER'
                  sh 'sudo ./upload_docker_jenkins.sh $USER $PASS'
               }
              }
             }


         stage('Associate with cluster') {
             steps {
                    withAWS(credentials:'aws-kubernetes') {
                        sh 'aws eks --region us-east-2 update-kubeconfig --name analytics-cluster'
                        sh 'kubectl get svc'

               }
              }
             }

//         removing deployment (changes LB address)
           stage('Delete previous deployment') {
             steps {
                    withAWS(credentials:'aws-kubernetes') {
                        sh 'kubectl get all -n stock-app'
                        sh 'kubectl delete -f deployment-config.yaml'
                        sh 'kubectl get all -n stock-app'
               }
              }
             }


         stage('Deploy app to k8s and check namespace') {
             steps {
                    withAWS(credentials:'aws-kubernetes') {
                        sh 'kubectl apply -f deployment-config.yaml'
                        sh 'sleep 60'
                        sh 'kubectl get all -n stock-app'
                        sh 'kubectl describe pods -n stock-app'
               }
              }
             }

         stage('Verify LB address') {
             steps {
                    withAWS(credentials:'aws-kubernetes') {
                        sh 'sleep 60'
                        sh 'kubectl get ingress/ingress-stock-app -n stock-app'
               }
              }
             }


         stage('Clean-up docker images') {
             steps {
                    sh 'df -h'
                    sh 'sudo docker system prune -f -a'
                    sh 'df -h'
              }
             }




         }
     }
