// src/SolutionDisplay.js
import React from "react";
// import "./SolutionDisplay.css"; // Make sure to create or update this CSS file

// Helper to add className based on result content (remains the same)
const getResultClassName = (result) => {
  if (!result) return "";
  const lowerResult = result.toLowerCase();
  if (
    lowerResult.includes("error") ||
    lowerResult.includes("fail") ||
    lowerResult.includes("could not determine")
  )
    return "result-error";
  if (lowerResult.includes("indeterminate") || lowerResult.includes("nan"))
    return "result-indeterminate";
  if (
    lowerResult.includes("dne") ||
    lowerResult.includes("differ") ||
    lowerResult.includes("oscillation")
  )
    return "result-dne";
  if (lowerResult.includes("∞") || lowerResult.includes("infinity"))
    return "result-infinity";
  return "result-success"; // Default class for finite results
};

function SolutionDisplay({ steps, result }) {
  // The 'result' prop here holds the final_result string from the API response
  if (!steps || steps.length === 0) {
    return null; // Don't render anything if there are no steps yet
  }

  // Find the actual final result step to avoid displaying it twice
  const finalStepIndex = steps.findIndex(
    (step) => step.method === "Final Result"
  );
  const stepsToDisplay =
    finalStepIndex !== -1 ? steps.slice(0, finalStepIndex) : steps;

  return (
    <div className="solution-display">
      <h2>Solution Steps</h2>
      <ul className="solution-steps">
        {stepsToDisplay.map((step, index) => (
          <li
            key={index}
            className={`step-item step-method-${step.method
              .toLowerCase()
              .replace(/[^a-z0-9]/g, "-")}`}
          >
            {/* *** CHANGE 1: Use step.method for the title *** */}
            <strong className="step-method">{step.method}:</strong>

            {/* *** CHANGE 2: Display step.description *** */}
            <p className="step-description">{step.description}</p>

            {/* *** CHANGE 3: Conditionally display expressions and step result *** */}
            {step.expression_before && (
              <div className="step-expression">
                <span className="expression-label">Before:</span>
                {/* Use <pre> for potentially multi-line pretty-printed expressions */}
                <pre className="expression-math">{step.expression_before}</pre>
              </div>
            )}
            {step.expression_after && (
              <div className="step-expression">
                <span className="expression-label">After:</span>
                <pre className="expression-math">{step.expression_after}</pre>
              </div>
            )}
            {/* Only show step.result if it's not the 'Final Result' step itself */}
            {step.result && step.method !== "Final Result" && (
              <div className="step-result">
                <span className="result-label">Step Result:</span>
                {/* Display step result inline or in pre, depending on expected content */}
                <span className="result-value">{step.result}</span>
              </div>
            )}
          </li>
        ))}
      </ul>

      {/* *** Final Result Section (uses the 'result' prop passed from App.js) *** */}
      {result && (
        <div className="final-result-container">
          <h3>Final Limit:</h3>
          <div className={`final-result-value ${getResultClassName(result)}`}>
            {/* Display final result - use <pre> if it might be complex/pretty */}
            <pre>{result}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default SolutionDisplay;
