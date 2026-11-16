pipeline {
    agent any

    environment {
        APP_NAME = "f1-predictor"
        IMAGE_TAG = "latest"
        EC2_HOST = "16.112.226.25"
        EC2_USER = "ubuntu"
        PORT = "8000"
    }

    stages {
        stage('1. Code Checkout') {
            steps {
                echo 'Pulling latest code from GitHub...'
                checkout scm
            }
        }

        stage('2. Automated Testing') {
            steps {
                echo 'Running Pytest Quality Gate...'
                sh '''
                    python3 -m venv .venv || true
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pytest tests/ -v
                '''
            }
        }

        stage('3. Build Verification') {
            steps {
                echo 'Validating Dockerfile and application structure...'
                sh '''
                    test -f Dockerfile
                    test -f requirements.txt
                    test -d app
                    echo "Application structure verified successfully."
                '''
            }
        }

        stage('4. Deploy to AWS EC2') {
            steps {
                echo "Deploying containerized ML application to AWS EC2 (${EC2_HOST})..."
                sshagent(['ec2-ssh-key']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} "
                            if [ ! -d ~/f1-ml-devops ]; then
                                git clone https://github.com/atharv-1201/f1-ml-devops.git ~/f1-ml-devops
                            fi
                            cd ~/f1-ml-devops
                            git fetch --all
                            git reset --hard origin/main
                            echo 'Building Docker container on EC2...'
                            docker build -t ${APP_NAME}:${IMAGE_TAG} .
                            echo 'Restarting service...'
                            docker stop ${APP_NAME} || true
                            docker rm ${APP_NAME} || true
                            docker run -d --name ${APP_NAME} --restart always -p ${PORT}:8000 ${APP_NAME}:${IMAGE_TAG}
                            sleep 5
                            curl -f http://localhost:${PORT}/health
                        "
                    '''
                }
            }
        }
    }

    post {
        success {
            echo " Pipeline Success: ML Inference live on AWS EC2 at http://${EC2_HOST}:${PORT}"
        }
        failure {
            echo " Pipeline Failed: Deployment was aborted."
        }
    }
}
