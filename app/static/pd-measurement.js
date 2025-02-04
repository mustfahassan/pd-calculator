// pd-measurement.js
class PDMeasurement {
    constructor() {
        this.videoElement = null;
        this.canvasElement = null;
        this.ctx = null;
        this.faceMesh = null;
        this.camera = null;
        this.isRunning = false;
        this.stableFrameCount = 0;
        this.STABILITY_THRESHOLD = 30;
        this.frameWidth = 640;
        this.frameHeight = 480;
        this.positionHistory = [];
        this.historySize = 5;
        this.measurementInProgress = false;
        
        // Face guide dimensions
        this.faceWidthRatio = 0.35;
        this.faceHeightRatio = 0.65;
        this.positionTolerance = 0.05;
        this.sizeTolerance = 0.15;

        this.LEFT_IRIS = [474, 475, 476, 477];
        this.RIGHT_IRIS = [469, 470, 471, 472];
    }

    async initialize() {
        console.log("Initializing PD Measurement...");
        try {
            // Create and setup video element
            this.videoElement = document.createElement('video');
            this.videoElement.setAttribute('playsinline', true);
            
            // Create and setup canvas
            this.canvasElement = document.createElement('canvas');
            this.canvasElement.width = this.frameWidth;
            this.canvasElement.height = this.frameHeight;
            this.ctx = this.canvasElement.getContext('2d');
            
            // Add canvas to video container
            const container = document.querySelector('.video-container');
            container.appendChild(this.canvasElement);
            
            // Initialize FaceMesh
            this.faceMesh = new FaceMesh({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
                }
            });

            // Configure FaceMesh
            await this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            // Set up FaceMesh callback
            this.faceMesh.onResults(async (results) => {
                await this.onResults(results);
            });

            console.log("Initialization complete. Starting camera...");
            return true;
        } catch (error) {
            console.error("Initialization error:", error);
            throw error;
        }
    }

    async startCamera() {
        try {
            console.log("Requesting camera access...");
            const constraints = {
                video: {
                    width: this.frameWidth,
                    height: this.frameHeight,
                    facingMode: 'user'
                }
            };
            
            this.camera = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.camera;
            
            // Wait for video to be ready
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    resolve();
                };
            });
            
            await this.videoElement.play();
            this.isRunning = true;
            this.processFrames();
            
            console.log("Camera started successfully");
            return true;
        } catch (error) {
            console.error("Camera start error:", error);
            throw error;
        }
    }

    async processFrames() {
        if (!this.isRunning) return;

        try {
            await this.faceMesh.send({image: this.videoElement});
            requestAnimationFrame(() => this.processFrames());
        } catch (error) {
            console.error("Frame processing error:", error);
        }
    }

    drawFaceGuide() {
        const centerX = this.frameWidth / 2;
        const centerY = this.frameHeight / 2;
        
        // Calculate dimensions based on golden ratio and face proportions
        const faceWidth = this.frameWidth * this.faceWidthRatio;
        const faceHeight = this.frameHeight * this.faceHeightRatio;
        
        // Set style for a softer, more natural look
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        this.ctx.lineWidth = 2;
        
        // Draw the face guide
        this.ctx.beginPath();
        
        // Calculate key points
        const topY = centerY - faceHeight/2;
        const bottomY = centerY + faceHeight/2;
        const leftX = centerX - faceWidth/2;
        const rightX = centerX + faceWidth/2;
        
        // Start from left side top
        this.ctx.moveTo(leftX + faceWidth/6, topY + faceHeight/6);
        
        // Top of head curve
        this.ctx.bezierCurveTo(
            leftX + faceWidth/3, topY,
            rightX - faceWidth/3, topY,
            rightX - faceWidth/6, topY + faceHeight/6
        );
        
        // Right side curve
        this.ctx.bezierCurveTo(
            rightX, topY + faceHeight/3,
            rightX, bottomY - faceHeight/3,
            rightX - faceWidth/4, bottomY - faceHeight/8
        );
        
        // Bottom curve (chin area)
        this.ctx.bezierCurveTo(
            rightX - faceWidth/3, bottomY,
            leftX + faceWidth/3, bottomY,
            leftX + faceWidth/4, bottomY - faceHeight/8
        );
        
        // Left side curve
        this.ctx.bezierCurveTo(
            leftX, bottomY - faceHeight/3,
            leftX, topY + faceHeight/3,
            leftX + faceWidth/6, topY + faceHeight/6
        );
        
        // Draw the outline
        this.ctx.stroke();
    }

    checkPositionStability(faceCenter) {
        if (this.positionHistory.length >= this.historySize) {
            this.positionHistory.shift();
        }
        this.positionHistory.push(faceCenter);
        
        if (this.positionHistory.length < this.historySize) return false;
        
        const xCoords = this.positionHistory.map(pos => pos[0]);
        const yCoords = this.positionHistory.map(pos => pos[1]);
        const xVariance = Math.max(...xCoords) - Math.min(...xCoords);
        const yVariance = Math.max(...yCoords) - Math.min(...yCoords);
        
        return xVariance < this.frameWidth * 0.03 && yVariance < this.frameHeight * 0.03;
    }

    getFaceBoundingBox(landmarks) {
        const coords = [];
        for (const landmark of landmarks) {
            coords.push({
                x: landmark.x * this.frameWidth,
                y: landmark.y * this.frameHeight
            });
        }
        
        return {
            left: Math.min(...coords.map(c => c.x)),
            right: Math.max(...coords.map(c => c.x)),
            top: Math.min(...coords.map(c => c.y)),
            bottom: Math.max(...coords.map(c => c.y))
        };
    }

    checkAlignment(landmarks) {
        const faceBox = this.getFaceBoundingBox(landmarks);
        const faceCenter = [
            (faceBox.left + faceBox.right) / 2,
            (faceBox.top + faceBox.bottom) / 2
        ];
        
        const centerX = this.frameWidth / 2;
        const centerY = this.frameHeight / 2;
        
        const xOffset = Math.abs(faceCenter[0] - centerX) / this.frameWidth;
        const yOffset = Math.abs(faceCenter[1] - centerY) / this.frameHeight;
        
        const faceWidth = faceBox.right - faceBox.left;
        const ovalWidth = this.frameWidth * this.faceWidthRatio;
        const widthDiff = Math.abs(faceWidth - ovalWidth) / ovalWidth;
        
        const isAligned = xOffset <= this.positionTolerance && 
                         yOffset <= this.positionTolerance && 
                         widthDiff <= this.sizeTolerance;
                         
        let message = "Center your face";
        if (widthDiff > this.sizeTolerance) {
            message = faceWidth < ovalWidth ? "Move closer" : "Move back";
        } else if (isAligned) {
            message = "Perfect";
        }
        
        return { 
            aligned: isAligned && this.checkPositionStability(faceCenter),
            message: message
        };
    }

    drawInstructions(message) {
        const y = this.frameHeight - 40;
        
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(0, y - 30, this.frameWidth, 60);
        
        this.ctx.fillStyle = 'white';
        this.ctx.font = '20px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(message, this.frameWidth / 2, y);
    }

    drawPupils(landmarks) {
        // Calculate iris centers
        const leftIrisPoints = this.LEFT_IRIS.map(index => ({
            x: landmarks[index].x * this.frameWidth,
            y: landmarks[index].y * this.frameHeight
        }));
        const rightIrisPoints = this.RIGHT_IRIS.map(index => ({
            x: landmarks[index].x * this.frameWidth,
            y: landmarks[index].y * this.frameHeight
        }));
    
        // Calculate centroids
        const leftCenter = {
            x: leftIrisPoints.reduce((sum, point) => sum + point.x, 0) / 4,
            y: leftIrisPoints.reduce((sum, point) => sum + point.y, 0) / 4
        };
        const rightCenter = {
            x: rightIrisPoints.reduce((sum, point) => sum + point.x, 0) / 4,
            y: rightIrisPoints.reduce((sum, point) => sum + point.y, 0) / 4
        };
    
        // Draw circles for iris centers and line
        this.ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        this.ctx.lineWidth = 2;
    
        // Draw left iris
        this.ctx.beginPath();
        this.ctx.arc(leftCenter.x, leftCenter.y, 3, 0, 2 * Math.PI);
        this.ctx.stroke();
    
        // Draw right iris
        this.ctx.beginPath();
        this.ctx.arc(rightCenter.x, rightCenter.y, 3, 0, 2 * Math.PI);
        this.ctx.stroke();
    
        // Draw connecting line
        this.ctx.beginPath();
        this.ctx.moveTo(leftCenter.x, leftCenter.y);
        this.ctx.lineTo(rightCenter.x, rightCenter.y);
        this.ctx.stroke();
    }
    
    // Add countdown method
    async startCountdown() {
        return new Promise(async (resolve) => {
            const countdownOverlay = document.getElementById('countdownOverlay');
            const countdownSpan = document.getElementById('countdown');
            
            countdownOverlay.style.display = 'flex';
            
            for (let i = 3; i > 0; i--) {
                countdownSpan.textContent = i;
                await new Promise(r => setTimeout(r, 1000));
            }
            
            countdownOverlay.style.display = 'none';
            resolve();
        });
    }
    
    // Update onResults method to include pupil drawing
    async onResults(results) {
        if (!this.ctx || !this.isRunning) return;
        
        // Clear canvas
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.frameWidth, this.frameHeight);
        
        // Save the current context state
        this.ctx.save();
        
        // Flip horizontally
        this.ctx.scale(-1, 1);
        this.ctx.translate(-this.frameWidth, 0);
        
        // Draw camera feed
        this.ctx.drawImage(this.videoElement, 0, 0, this.frameWidth, this.frameHeight);
        
        // Restore the context to draw overlays normally
        this.ctx.restore();
        
        // Draw face guide
        this.drawFaceGuide();
        
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            
            // Transform landmarks to account for flipped view
            const flippedLandmarks = landmarks.map(landmark => ({
                ...landmark,
                x: 1 - landmark.x // Flip x coordinates
            }));
            
            const alignment = this.checkAlignment(flippedLandmarks);
            
            // Draw pupils with flipped coordinates
            this.drawPupils(flippedLandmarks);
            
            if (alignment.aligned) {
                this.stableFrameCount++;
                if (this.stableFrameCount >= this.STABILITY_THRESHOLD && !this.measurementInProgress) {
                    await this.startMeasurement(flippedLandmarks);
                }
            } else {
                this.stableFrameCount = 0;
            }
            
            // Update progress bar
            const progressBar = document.getElementById('stabilityProgress');
            if (progressBar) {
                const progress = (this.stableFrameCount / this.STABILITY_THRESHOLD) * 100;
                progressBar.style.width = `${Math.min(progress, 100)}%`;
            }
            
            this.drawInstructions(alignment.message);
        } else {
            this.drawInstructions("No face detected");
            this.stableFrameCount = 0;
        }
    }
    
    // Update startMeasurement method
    async startMeasurement(landmarks) {
        if (this.measurementInProgress) return;
        
        try {
            this.measurementInProgress = true;
            
            // Pause face processing
            const wasRunning = this.isRunning;
            this.isRunning = false;
            
            // Start countdown
            await this.startCountdown();
    
            // Convert landmarks to the format expected by the backend
            const landmarkData = {};
            landmarks.forEach((landmark, index) => {
                landmarkData[index] = {
                    x: landmark.x,
                    y: landmark.y,
                    z: landmark.z
                };
            });
    
            // Send landmarks to backend
            const response = await fetch('/calculate_pd', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ landmarks: landmarkData })
            });
    
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayResults(data);
                // Stop camera and update UI
                this.stop();
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('measureBtn').style.display = 'block';
                document.getElementById('measureBtn').textContent = '📷 Measure Again';
                document.getElementById('measureBtn').disabled = false;
            } else {
                console.error('Measurement failed:', data.message);
                // Resume face processing if measurement fails
                this.isRunning = wasRunning;
                this.processFrames();
            }
        } catch (error) {
            console.error('Error during measurement:', error);
            this.isRunning = true;
            this.processFrames();
        } finally {
            this.measurementInProgress = false;
        }
    }

    displayResults(data) {
        const pdValue = document.getElementById('pdValue');
        const confidenceValue = document.getElementById('confidenceValue');
        const resultsCard = document.getElementById('resultsCard');
        
        // Round PD value based on decimal part
        const pdNumber = parseFloat(data.pd_mm);
        const decimalPart = pdNumber % 1;
        const roundedPD = decimalPart >= 0.5 ? Math.ceil(pdNumber) : Math.floor(pdNumber);
        
        // Display rounded PD value
        pdValue.textContent = `${roundedPD} mm`;
        confidenceValue.textContent = `${Math.round(data.confidence)}%`;
        
        confidenceValue.className = 'measurement-value';
        if (data.confidence >= 90) {
            confidenceValue.classList.add('success');
        } else if (data.confidence >= 75) {
            confidenceValue.classList.add('good');
        } else {
            confidenceValue.classList.add('warning');
        }
        
        resultsCard.style.display = 'block';
    }

    stop() {
        this.isRunning = false;
        if (this.camera) {
            this.camera.getTracks().forEach(track => track.stop());
        }
        if (this.canvasElement) {
            this.canvasElement.remove();
        }
        this.stableFrameCount = 0;
        this.positionHistory = [];
        this.measurementInProgress = false;
    }
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded, setting up PD measurement...");
    const pdMeasurement = new PDMeasurement();

    const measureBtn = document.getElementById('measureBtn');
    const stopBtn = document.getElementById('stopBtn');

    measureBtn.addEventListener('click', async () => {
        console.log("Start button clicked");
        try {
            measureBtn.disabled = true;
            await pdMeasurement.initialize();
            await pdMeasurement.startCamera();
            measureBtn.style.display = 'none';
            stopBtn.style.display = 'block';
        } catch (error) {
            console.error("Failed to start measurement:", error);
            measureBtn.disabled = false;
            alert("Failed to start camera. Please ensure camera permissions are granted and try again.");
        }
    });

    stopBtn.addEventListener('click', () => {
        pdMeasurement.stop();
        stopBtn.style.display = 'none';
        measureBtn.style.display = 'block';
        measureBtn.disabled = false;
    });
});