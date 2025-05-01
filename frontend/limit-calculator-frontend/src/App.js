// src/App.js
import React, { useState } from "react";
import axios from "axios"; // Using axios
import LimitInputForm from "./LimitInputForm"; // Assuming this component exists
import SolutionDisplay from "./SolutionDisplay"; // Assuming this component exists
import "./App.css"; // Assuming basic styling exists

// IMPORTANT: Replace with your actual backend URL if different
// Ensure your FastAPI backend allows requests from your frontend's origin (CORS)
const API_BASE_URL = "http://localhost:8000"; // Or http://127.0.0.1:8000

function App() {
  const [expression, setExpression] = useState("sin(x)/x"); // Default example
  const [variable, setVariable] = useState("x");
  const [tendingTo, setTendingTo] = useState("0");
  const [steps, setSteps] = useState([]);
  const [result, setResult] = useState(null); // This will store the final_result string
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCalculateLimit = async () => {
    setIsLoading(true);
    setError(null); // Clear previous errors
    setSteps([]); // Clear previous steps
    setResult(null); // Clear previous result

    try {
      // *** CHANGE 1: Update the API endpoint URL ***
      const response = await axios.post(
        `${API_BASE_URL}/limit_steps`, // Added '/mylimits' prefix
        {
          expression: expression,
          variable: variable,
          tending_to: tendingTo,
        }
      );

      // *** CHANGE 2: Check for 'final_result' instead of 'result' ***
      // Check if the response structure is as expected from LimitResponseSteps
      if (
        response.data &&
        Array.isArray(response.data.steps) &&
        typeof response.data.final_result === "string" // Check for final_result
      ) {
        setSteps(response.data.steps);
        // *** CHANGE 3: Set result state using 'final_result' ***
        setResult(response.data.final_result); // Use final_result from response
      } else {
        console.error("Unexpected response structure:", response.data);
        setError("Received an unexpected response format from the server.");
      }
    } catch (err) {
      console.error("API Error:", err);
      if (err.response) {
        // Server responded with non-2xx status
        console.error("Error Response Data:", err.response.data);
        console.error("Error Response Status:", err.response.status);
        // Use the detail message from FastAPI if available
        setError(
          err.response.data?.detail ||
            `Server Error: ${err.response.status}. Check input syntax or server logs.`
        );
      } else if (err.request) {
        // Request made but no response received (Network error, server down?)
        console.error("Error Request:", err.request);
        setError(
          "Could not connect to the calculation server. Is it running and accessible?"
        );
      } else {
        // Setup error before request was sent
        console.error("Error Message:", err.message);
        setError(`An error occurred: ${err.message}`);
      }
      setSteps([]); // Clear steps on error
      setResult(null); // Clear result on error
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Advanced Limit Calculator</h1>
        <p>
          Enter expression using SymPy syntax (e.g., `x**2`, `sin(x)`, `exp(x)`,
          `pi`, `oo`)
        </p>
      </header>

      <LimitInputForm
        expression={expression}
        variable={variable}
        tendingTo={tendingTo}
        onExpressionChange={(e) => setExpression(e.target.value)}
        onVariableChange={(e) => setVariable(e.target.value)}
        onTendingToChange={(e) => setTendingTo(e.target.value)}
        onSubmit={handleCalculateLimit}
        isLoading={isLoading}
      />

      {isLoading && (
        <div className="loading-indicator">Calculating limit...</div>
      )}

      {error && <div className="error-message">Error: {error}</div>}

      {/* Only render SolutionDisplay if not loading, no error, and steps exist */}
      {/* The 'result' state now correctly holds the 'final_result' from API */}
      {!isLoading && !error && steps.length > 0 && (
        <SolutionDisplay steps={steps} result={result} />
      )}
    </div>
  );
}

export default App;
