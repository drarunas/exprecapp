// Import the Firebase functions you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-analytics.js";
import { getAuth, signInWithPopup, GoogleAuthProvider, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js";

// Initialize Firebase app
const firebaseConfig = {
  apiKey: "AIzaSyA6928B6iVjkwYUf7l6_d1wKz6QoTMOyME",
  authDomain: "exrecapp-7deb8.firebaseapp.com",
  projectId: "exrecapp-7deb8",
  storageBucket: "exrecapp-7deb8.appspot.com",
  messagingSenderId: "1036215512579",
  appId: "1:1036215512579:web:b749e5b288fc9d75ef6836",
  measurementId: "G-7B14G4ZFLX"
};


// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// Initialize Firebase Authentication
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Function to display user info
const displayUserInfo = (user) => {
  document.getElementById('user-name').textContent = user.displayName;
  document.getElementById('user-email').textContent = user.email;
  document.getElementById('user-photo').src = user.photoURL;

  // Show the user info section and hide the login button
  document.getElementById('user-info').style.display = 'block';
  document.getElementById('google-login-btn').style.display = 'none';
  
};

// Check if user is already logged in
onAuthStateChanged(auth, (user) => {
  if (user) {
      // User is signed in, display their info
      displayUserInfo(user);
      document.getElementById('google-logout-btn').style.display = 'block';
  } else {
      // No user is signed in, show the login button
      document.getElementById('google-login-btn').style.display = 'block';
      document.getElementById('user-info').style.display = 'none';
      document.getElementById('google-logout-btn').style.display = 'none';
  }
});

// Function to handle logout
const handleLogout = () => {
  signOut(auth).then(() => {
      // Hide user info and show login button
      document.getElementById('user-info').style.display = 'none';
      document.getElementById('google-login-btn').style.display = 'block';
      console.log('User signed out.');
  }).catch((error) => {
      console.error('Error during logout:', error);
  });
};


// Function to handle Google sign-in
const loginWithGoogle = () => {
  signInWithPopup(auth, provider)
      .then((result) => {
          const user = result.user;
          // Display user info
          displayUserInfo(user);
      })
      .catch((error) => {
          console.error('Error during login:', error);
      });
};

// Add event listener to the login button
document.getElementById('google-login-btn').addEventListener('click', loginWithGoogle);
document.getElementById('google-logout-btn').addEventListener('click', handleLogout);
