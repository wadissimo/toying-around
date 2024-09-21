import { useEffect } from "react";
import { useAuth } from "../contexts/FakeAuthContext";
import styles from "./User.module.css";
import { useNavigate } from "react-router-dom";

const FAKE_USER = {
  name: "Jack",
  email: "jack@example.com",
  password: "qwerty",
};
const avatar = "https://i.pravatar.cc/100?u=zz";

function User() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  if (!isAuthenticated) return <></>;

  function handleClick() {
    logout();
    navigate("/login");
  }

  return (
    <div className={styles.user}>
      <span>Welcome, {user}</span>
      <button onClick={handleClick}>Logout</button>
    </div>
  );
}

export default User;

/*
CHALLENGE

1) Add `AuthProvider` to `App.jsx`
2) In the `Login.jsx` page, call `login()` from context
3) Inside an effect, check whether `isAuthenticated === true`. If so, programatically navigate to `/app`
4) In `User.js`, read and display logged in user from context (`user` object). Then include this component in `AppLayout.js`
5) Handle logout button by calling `logout()` and navigating back to `/`
*/
