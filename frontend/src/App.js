import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend } from 'chart.js';
import './App.css';

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend);

const API_URL = 'http://localhost:8000/api';

const DIAGNOSIS_TRANSLATIONS = {
  'MildDemented': 'Легкая степень',
  'ModerateDemented': 'Умеренная степень',
  'NonDemented': 'Норма',
  'VeryMildDemented': 'Очень легкая степень',
};

// Генерация ключа для кэша на основе файла
const generateCacheKey = (file) => {
  return `mri_analysis_${file.name}_${file.size}`;
};

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [gradcamOpacity, setGradcamOpacity] = useState(0.5);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: 'image/*',
    maxFiles: 1,
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
      setResults(null);
    }
  });

  const analyzeImage = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/analyze`, formData);
      setResults(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Ошибка при анализе изображения');
    } finally {
      setLoading(false);
    }
  };

  const classifyImage = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/classify`, formData);
      setResults({
        classification: response.data,
        interpretation: null
      });
    } catch (error) {
      console.error('Classification error:', error);
      alert('Ошибка при классификации изображения');
    } finally {
      setLoading(false);
    }
  };

  const renderDiagnosisChart = (probabilities) => {
    if (!probabilities || !chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const chartCtx = chartRef.current.getContext('2d');
    if (!chartCtx) return;
    
    const labels = Object.keys(DIAGNOSIS_TRANSLATIONS);
    const data = labels.map(label => probabilities[label]);
    
    chartInstance.current = new Chart(chartCtx, {
      type: 'bar',
      data: {
        labels: labels.map(key => DIAGNOSIS_TRANSLATIONS[key]),
        datasets: [{
          data: data,
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 206, 86, 0.7)'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
            ticks: {
              callback: value => `${(value * 100).toFixed(0)}%`
            }
          }
        },
        plugins: {
          legend: {
            display: false
          }
        }
      }
    });
  };

  useEffect(() => {
    if (results?.classification?.probabilities) {
      renderDiagnosisChart(results.classification.probabilities);
    }
  }, [results]);

  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  return (
    <div className="app-container">
      <div className="upload-section">
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <p>{image ? file.name : 'Перетащите МРТ-снимок сюда'}</p>
        </div>
        <div className="button-group">
          <button 
            onClick={analyzeImage} 
            disabled={!image || loading}
            className="analyze-btn"
          >
            {loading ? 'Анализ...' : 'Полный анализ'}
          </button>
          <button 
            onClick={classifyImage} 
            disabled={!image || loading}
            className="classify-btn"
          >
            {loading ? 'Классификация...' : 'Классифицировать'}
          </button>
        </div>
      </div>

      {results && (
        <div className="results-row">
          {results.interpretation && (
            <>
              {/* Блок Grad-CAM с оригинальным изображением */}
              <div className="gradcam-block">
                <div className="gradcam-container">
                  <img 
                    src={image} 
                    alt="Оригинальное изображение" 
                    className="original-underlay"
                  />
                  <img 
                    src={`data:image/png;base64,${results.interpretation.additional_info.heatmap_img}`} 
                    alt="Grad-CAM" 
                    className="heatmap-overlay"
                    style={{ opacity: gradcamOpacity }}
                  />
                </div>
                
                <div className="opacity-control">
                  <label>Прозрачность Grad-CAM:</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={gradcamOpacity}
                    onChange={(e) => setGradcamOpacity(e.target.value)}
                  />
                </div>
              </div>

              {/* Блок LIME */}
              {results.interpretation.additional_info.lime_img && (
                <div className="lime-block">
                  <div className="lime-container">
                    <img 
                      src={`data:image/png;base64,${results.interpretation.additional_info.lime_img}`} 
                      alt="LIME объяснение" 
                    />
                  </div>
                </div>
              )}
            </>
          )}

          {/* Блок графика и диагноза */}
          <div className="chart-block">
            <div className="chart-container">
              <canvas ref={chartRef} />
            </div>
            <div className="diagnosis-box">
              <h3>Результат:</h3>
              <p className="diagnosis">
                {DIAGNOSIS_TRANSLATIONS[results.classification.class_name] || results.classification.class_name}
              </p>
              <p className="confidence">
                Уверенность: {(results.classification.confidence * 100).toFixed(1)}%
              </p>
              {/* {results.interpretation && (
                <>
                  <h4>Находки:</h4>
                  <ul>
                    {results.interpretation.findings.map((finding, index) => (
                      <li key={index}>{finding}</li>
                    ))}
                  </ul>
                  <h4>Рекомендации:</h4>
                  <ul>
                    {results.interpretation.recommendations.map((recommendation, index) => (
                      <li key={index}>{recommendation}</li>
                    ))}
                  </ul>
                </>
              )} */}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;