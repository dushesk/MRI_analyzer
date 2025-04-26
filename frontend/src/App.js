import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Chart, BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend } from 'chart.js';
import 'react-medium-image-zoom/dist/styles.css';
import './App.css';

// Регистрация компонентов Chart.js
Chart.register(BarController, BarElement, LinearScale, CategoryScale, Tooltip, Legend);

const API_URL = 'http://localhost:8000/api';

// Перевод диагнозов
const DIAGNOSIS_TRANSLATIONS = {
  'MildDemented': 'Легкая',
  'ModerateDemented': 'Умеренная',
  'NonDemented': 'Норма',
  'VeryMildDemented': 'Очень легкая',

};

function App() {
  const [image, setImage] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [opacity, setOpacity] = useState(0.5);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const imageContainerRef = useRef(null);

  // Настройка dropzone для загрузки файлов
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: 'image/*',
    maxFiles: 1,
    onDrop: acceptedFiles => {
      const file = acceptedFiles[0];
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result);
      reader.readAsDataURL(file);
      setResults(null); // Сбрасываем предыдущие результаты
    }
  });

  // Анализ изображения
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
      alert('Ошибка при анализе изображения. Проверьте консоль для подробностей.');
    } finally {
      setLoading(false);
    }
  };

  // Отрисовка диаграммы
  const renderDiagnosisChart = (predictions) => {
    if (!predictions || !chartRef.current) return;

    // Удаляем предыдущий график, если он существует
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const chartCtx = chartRef.current.getContext('2d');
    if (!chartCtx) return;
    
    chartInstance.current = new Chart(chartCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(DIAGNOSIS_TRANSLATIONS),
        datasets: [{
          label: 'Вероятность (%)',
          data: predictions,
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(54, 162, 235, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 206, 86, 0.7)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(255, 206, 86, 1)'
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

  // Автоматическое обновление диаграммы при изменении результатов
  useEffect(() => {
    if (results?.predictions) {
      renderDiagnosisChart(results.predictions);
    }
  }, [results]);

  // Очистка при размонтировании компонента
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, []);

  return (
    <div className="compact-app">
      <header className="app-header">
        <h1>Анализатор МРТ</h1>
        <div {...getRootProps()} className={`upload-area ${isDragActive ? 'dragging' : ''}`}>
          <input {...getInputProps()} />
          {image ? (
            <p className="file-name">{file.name}</p>
          ) : (
            <p>{isDragActive ? 'Отпустите файл' : 'Перетащите снимок'}</p>
          )}
        </div>
        <button 
          onClick={analyzeImage} 
          disabled={!image || loading}
          className="analyze-button"
        >
          {loading ? 'Анализ...' : 'Анализировать'}
        </button>
      </header>

      {results && (
        <div className="results-area">
          <div className="visualization-section">
            <div className="image-comparison" ref={imageContainerRef}>
              <img src={image} alt="Оригинал" className="original-image" />
              <img 
                src={`data:image/png;base64,${results.heatmap_img}`} 
                alt="Heatmap" 
                className="heatmap-layer"
                style={{ opacity }}
              />
              <div className="zoom-controls">
                <button onClick={() => imageContainerRef.current.requestFullscreen()}>
                  🔍
                </button>
              </div>
              <div className="opacity-control">
                <span>Прозрачность:</span>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                />
              </div>
            </div>

            <div className="diagnosis-summary">
              <h3>Заключение: <strong>{DIAGNOSIS_TRANSLATIONS[results.predicted_class] || results.predicted_class}</strong></h3>
              <p>Вероятность: {(results.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>

          <div className="chart-section">
            <canvas ref={chartRef} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;