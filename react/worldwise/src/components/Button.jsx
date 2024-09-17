import React from "react";
import styles from "./Button.module.css";

export default function Button({ type, children, onClick }) {
  return (
    <button onClick={onClick} className={`${styles.btn} ${styles[type]}`}>
      {children}
    </button>
  );
}
