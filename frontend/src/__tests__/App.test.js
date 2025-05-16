import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import App from '../App';

// Mock axios
jest.mock('axios');

describe('App Component', () => {
    beforeEach(() => {
        // Reset all mocks before each test
        jest.clearAllMocks();
    });

    it('renders without crashing', () => {
        render(<App />);
        expect(screen.getByText(/MRI Analysis/i)).toBeInTheDocument();
    });

    it('handles image upload', async () => {
        render(<App />);
        
        // Create a test file
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        
        // Get the file input
        const input = screen.getByTestId('file-input');
        
        // Upload the file
        await userEvent.upload(input, file);
        
        // Check if the image preview is displayed
        expect(screen.getByTestId('image-preview')).toBeInTheDocument();
    });

    it('handles successful image analysis', async () => {
        // Mock successful API response
        const mockResponse = {
            data: {
                classification: {
                    class_name: 'NonDemented',
                    confidence: 0.95,
                    class_id: 2,
                    probabilities: {
                        MildDemented: 0.1,
                        ModerateDemented: 0.05,
                        NonDemented: 0.95,
                        VeryMildDemented: 0.1
                    }
                },
                interpretation: {
                    findings: ['Test finding'],
                    recommendations: ['Test recommendation'],
                    additional_info: {
                        heatmap_img: 'base64_encoded_image',
                        lime_img: 'base64_encoded_image'
                    }
                }
            }
        };
        
        axios.post.mockResolvedValueOnce(mockResponse);
        
        render(<App />);
        
        // Upload test image
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const input = screen.getByTestId('file-input');
        await userEvent.upload(input, file);
        
        // Click analyze button
        const analyzeButton = screen.getByText(/Analyze/i);
        fireEvent.click(analyzeButton);
        
        // Wait for results
        await waitFor(() => {
            expect(screen.getByText('NonDemented')).toBeInTheDocument();
            expect(screen.getByText('95%')).toBeInTheDocument();
        });
    });

    it('handles analysis error', async () => {
        // Mock API error
        axios.post.mockRejectedValueOnce(new Error('Analysis failed'));
        
        render(<App />);
        
        // Upload test image
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const input = screen.getByTestId('file-input');
        await userEvent.upload(input, file);
        
        // Click analyze button
        const analyzeButton = screen.getByText(/Analyze/i);
        fireEvent.click(analyzeButton);
        
        // Check for error message
        await waitFor(() => {
            expect(screen.getByText(/Error analyzing image/i)).toBeInTheDocument();
        });
    });

    it('handles invalid file type', async () => {
        render(<App />);
        
        // Try to upload non-image file
        const file = new File(['test'], 'test.txt', { type: 'text/plain' });
        const input = screen.getByTestId('file-input');
        await userEvent.upload(input, file);
        
        // Check for error message
        expect(screen.getByText(/Please upload an image file/i)).toBeInTheDocument();
    });

    it('handles image classification only', async () => {
        // Mock successful classification response
        const mockResponse = {
            data: {
                class_name: 'NonDemented',
                confidence: 0.95,
                class_id: 2,
                probabilities: {
                    MildDemented: 0.1,
                    ModerateDemented: 0.05,
                    NonDemented: 0.95,
                    VeryMildDemented: 0.1
                }
            }
        };
        
        axios.post.mockResolvedValueOnce(mockResponse);
        
        render(<App />);
        
        // Upload test image
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const input = screen.getByTestId('file-input');
        await userEvent.upload(input, file);
        
        // Click classify button
        const classifyButton = screen.getByText(/Classify/i);
        fireEvent.click(classifyButton);
        
        // Wait for results
        await waitFor(() => {
            expect(screen.getByText('NonDemented')).toBeInTheDocument();
            expect(screen.getByText('95%')).toBeInTheDocument();
        });
    });

    it('handles image interpretation only', async () => {
        // Mock successful interpretation response
        const mockResponse = {
            data: {
                findings: ['Test finding'],
                recommendations: ['Test recommendation'],
                additional_info: {
                    heatmap_img: 'base64_encoded_image',
                    lime_img: 'base64_encoded_image'
                }
            }
        };
        
        axios.post.mockResolvedValueOnce(mockResponse);
        
        render(<App />);
        
        // Upload test image
        const file = new File(['test'], 'test.png', { type: 'image/png' });
        const input = screen.getByTestId('file-input');
        await userEvent.upload(input, file);
        
        // Click interpret button
        const interpretButton = screen.getByText(/Interpret/i);
        fireEvent.click(interpretButton);
        
        // Wait for results
        await waitFor(() => {
            expect(screen.getByText('Test finding')).toBeInTheDocument();
            expect(screen.getByText('Test recommendation')).toBeInTheDocument();
        });
    });
}); 