import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import Zoom from 'react-medium-image-zoom';
import { Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend } from 'chart.js';
import 'react-medium-image-zoom/dist/styles.css';
import './App.css';

Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend);

const API_URL = 'http://localhost:8000/api';

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const chartRef = useRef(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: 'image/*',
    maxFiles: 1,
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
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
      renderDiagnosisChart(response.data.predictions);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Ошибка при анализе изображения. Проверьте консоль для подробностей.');
    } finally {
      setLoading(false);
    }
  };

  const renderDiagnosisChart = (predictions) => {
    if (chartRef.current?.chart) {
      chartRef.current.chart.destroy();
    }
    
    const chartCtx = chartRef.current?.getContext('2d');
    if (!chartCtx) return;
    
    chartRef.current.chart = new Chart(chartCtx, {
      type: 'bar',
      data: {
        labels: ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'],
        datasets: [{
          label: 'Вероятность диагноза (%)',
          data: predictions,
          backgroundColor: [
            'rgba(75, 192, 192, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(255, 99, 132, 0.7)'
          ],
          borderColor: [
            'rgba(75, 192, 192, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(255, 99, 132, 1)'
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
          tooltip: {
            callbacks: {
              label: context => `${context.dataset.label}: ${(context.raw * 100).toFixed(2)}%`
            }
          },
          legend: {
            display: false
          }
        }
      }
    });
  };

  useEffect(() => {
    return () => {
      if (chartRef.current?.chart) {
        chartRef.current.chart.destroy();
      }
    };
  }, []);

  const translateDiagnosis = (diagnosis) => {
    const translations = {
      'NonDemented': 'Норма',
      'VeryMildDemented': 'Очень легкая деменция',
      'MildDemented': 'Легкая деменция',
      'ModerateDemented': 'Умеренная деменция'
    };
    return translations[diagnosis] || diagnosis;
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Анализатор МРТ-изображений</h1>
      
      <div {...getRootProps()} className={`upload-zone ${isDragActive ? 'active' : ''} ${image ? 'has-image' : ''}`}>
        <input {...getInputProps()} />
        {image ? (
          <div className="image-preview">
            <Zoom zoomMargin={40}>
              <img src={image} alt="МРТ-изображение" className="mri-image"/>
            </Zoom>
          </div>
        ) : (
          <p className="upload-text">
            {isDragActive ? 'Отпустите для загрузки' : 'Перетащите МРТ-изображение или кликните для выбора'}
          </p>
        )}
      </div>

      <button 
        onClick={analyzeImage} 
        disabled={!image || loading}
        className={`analyze-btn ${loading ? 'loading' : ''} ${!image ? 'disabled' : ''}`}
      >
        {loading ? (
          <span className="loading-spinner">
            <svg className="spinner-icon" viewBox="0 0 24 24">
              <circle className="spinner-track" cx="12" cy="12" r="10"/>
              <path className="spinner-path" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
            </svg>
            Идет анализ...
          </span>
        ) : 'Проанализировать'}
      </button>

      {results && (
        <div className="results-container">
          <h2 className="results-title">Результаты анализа</h2>
          
          <div className="result-section">
            <h3>Области интереса (Grad-CAM)</h3>
            <div className="heatmap-container">
              <img 
                src={`data:image/png;base64,${results.heatmap_img}`} 
                alt="Heatmap" 
                className="heatmap-image"
              />
            </div>
          </div>
          
          <div className="result-section">
            <h3>Вероятность диагноза</h3>
            <div className="chart-container">
              <canvas ref={chartRef} className="diagnosis-chart"/>
            </div>
          </div>
          
          <div className="diagnosis-conclusion">
            <h3>Заключение:</h3>
            <p className="diagnosis-text">
              {translateDiagnosis(results.predicted_class)} (вероятность: {(results.confidence * 100).toFixed(2)}%)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;