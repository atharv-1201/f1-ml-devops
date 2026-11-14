pipeline {
    agent any

    environment {
        APP_NAME = "f1-predictor"
        IMAGE_TAG = "latest"
        EC2_HOST = "18.61.27.74"
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
                echo 'Running Unit Tests and Quality Checks...'
                sh '''
                    python3 -m venv .venv || true
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pytest tests/ -v
                '''
            }
        }

        stage('3. Build Docker Container') {
            steps {
                echo 'Building Docker Image...'
                sh 'docker build -t ${APP_NAME}:${IMAGE_TAG} .'
            }
        }

        stage('4. Deploy to AWS EC2') {
            steps {
                echo 'Deploying containerized ML application to AWS EC2...'
                sshagent(['ec2-ssh-key']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} "
                            if [ ! -d ~/devops-project ]; then
                                git clone https://github.com/${GITHUB_REPOSITORY}.git ~/devops-project
                            fi
                            cd ~/devops-project
                            git pull
                            docker build -t ${APP_NAME}:${IMAGE_TAG} .
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
            echo "✅ Pipeline Success: ML Inference live on AWS EC2 at http://${EC2_HOST}:${PORT}"
        }
        failure {
            echo "❌ Pipeline Failed: Deployment was aborted."
        }
    }
}
