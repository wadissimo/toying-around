import { useNavigate } from "react-router-dom";
import Map from "../components/Map";
import Sidebar from "../components/Sidebar";
import User from "../components/User";
import { useAuth } from "../contexts/FakeAuthContext";
import styles from "./AppLayout.module.css";
import { useEffect } from "react";

export default function AppLayout() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  useEffect(
    function () {
      if (!isAuthenticated) {
        navigate("/login");
      }
    },
    [navigate, isAuthenticated]
  );
  return (
    <div className={styles.app}>
      <User />
      <Sidebar />
      <Map />
    </div>
  );
}
