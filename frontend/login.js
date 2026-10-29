(function () {
  "use strict";

  // If a session is already active, skip straight to the dashboard.
  if (localStorage.getItem("isLoggedIn") === "true") {
    window.location.replace("dashboard/index.html");
  }

  var DEMO_OTP = "426190";
  var currentView = "sign-in";
  var countdownValue = 45;
  var countdownTimer = null;
  var otpInputs = Array.prototype.slice.call(
    document.querySelectorAll(".otp-input"),
  );

  var views = {
    "sign-in": document.getElementById("view-sign-in"),
    register: document.getElementById("view-register"),
    forgot: document.getElementById("view-forgot"),
    otp: document.getElementById("view-otp"),
  };

  function setView(nextView) {
    currentView = nextView;
    Object.keys(views).forEach(function (viewName) {
      views[viewName].hidden = viewName !== nextView;
    });
    clearErrors();
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (nextView === "otp") {
      startCountdown();
      window.setTimeout(function () {
        if (otpInputs[0]) otpInputs[0].focus();
      }, 100);
    } else {
      stopCountdown();
    }
  }

  function clearErrors() {
    document.querySelectorAll(".field-error").forEach(function (error) {
      error.textContent = "";
    });
  }

  function showError(name, message) {
    var error = document.querySelector('[data-error-for="' + name + '"]');
    if (error) error.textContent = message;
  }

  function showOtpError(message) {
    var error = document.getElementById("otp-error");
    if (error) error.textContent = message;
  }

  function showNotice(message) {
    var notice = document.getElementById("sign-in-notice");
    notice.textContent = "✓  " + message;
    notice.hidden = false;
  }

  function resetNotice() {
    var notice = document.getElementById("sign-in-notice");
    notice.textContent = "";
    notice.hidden = true;
  }

  function startCountdown() {
    stopCountdown();
    countdownValue = 45;
    updateCountdown();
    document.getElementById("resend-button").disabled = true;
    document.querySelector(".otp-options").classList.remove("expired");
    countdownTimer = window.setInterval(function () {
      countdownValue -= 1;
      updateCountdown();
      if (countdownValue <= 0) {
        stopCountdown();
        document.getElementById("resend-button").disabled = false;
        document.querySelector(".otp-options").classList.add("expired");
      }
    }, 1000);
  }

  function stopCountdown() {
    if (countdownTimer) {
      window.clearInterval(countdownTimer);
      countdownTimer = null;
    }
  }

  function updateCountdown() {
    var countdown = document.getElementById("countdown");
    if (countdownValue <= 0) {
      countdown.textContent = "Code expired";
      return;
    }
    countdown.textContent =
      "Code expires in 00:" + String(countdownValue).padStart(2, "0");
  }

  function value(form, name) {
    var field = form.elements[name];
    return field ? String(field.value).trim() : "";
  }

  document.querySelectorAll("[data-open-view]").forEach(function (button) {
    button.addEventListener("click", function () {
      resetNotice();
      setView(button.getAttribute("data-open-view"));
    });
  });

  document.querySelectorAll("[data-password-toggle]").forEach(function (button) {
    button.addEventListener("click", function () {
      var input = document.getElementById(
        button.getAttribute("data-password-toggle"),
      );
      if (!input) return;
      var showing = input.type === "text";
      input.type = showing ? "password" : "text";
      button.setAttribute("aria-label", showing ? "Show password" : "Hide password");
      button.textContent = showing ? "◉" : "◌";
    });
  });

  document
    .getElementById("sign-in-form")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      clearErrors();
      resetNotice();
      var identifier = value(event.target, "identifier");
      var password = value(event.target, "password");
      var valid = true;

      if (!identifier) {
        showError("identifier", "Enter your email, Gmail, or phone number.");
        valid = false;
      }
      if (!password) {
        showError("password", "Enter your password to continue.");
        valid = false;
      }
      if (valid) {
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('userIdentifier', identifier);
        // Redirect user to the dashboard
        window.location.href = 'dashboard/index.html';
      } else {
          alert('Please enter your credentials');
      }
    });

  document
    .getElementById("register-form")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      clearErrors();
      var form = event.target;
      var requiredFields = [
        "username",
        "fullName",
        "phone",
        "email",
        "dateOfBirth",
        "interests",
        "password",
        "confirmPassword",
      ];
      var valid = true;

      requiredFields.forEach(function (fieldName) {
        if (!value(form, fieldName)) {
          showError(fieldName, "This field is required.");
          valid = false;
        }
      });

      var password = value(form, "password");
      var confirmPassword = value(form, "confirmPassword");
      if (password && password.length < 8) {
        showError("register-password", "Use at least 8 characters.");
        valid = false;
      }
      if (password && confirmPassword && password !== confirmPassword) {
        showError("confirmPassword", "Passwords do not match.");
        valid = false;
      }

      if (!valid) return;

      var email = value(form, "email");
      form.reset();
      document.getElementById("login-identifier").value = email;
      setView("sign-in");
      showNotice("Account created. Sign in to open your VeritiScan workspace.");
    });

  document
    .getElementById("forgot-form")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      clearErrors();
      var identifier = value(event.target, "resetIdentifier");
      if (!identifier) {
        showError("resetIdentifier", "Enter the email or phone linked to your account.");
        return;
      }
      document.getElementById("otp-recipient").textContent = identifier;
      otpInputs.forEach(function (input) {
        input.value = "";
      });
      setView("otp");
    });

  otpInputs.forEach(function (input, index) {
    input.addEventListener("input", function () {
      input.value = input.value.replace(/\D/g, "").slice(-1);
      showOtpError("");
      if (input.value && otpInputs[index + 1]) otpInputs[index + 1].focus();
    });

    input.addEventListener("keydown", function (event) {
      if (event.key === "Backspace" && !input.value && otpInputs[index - 1]) {
        otpInputs[index - 1].focus();
      }
      if (event.key === "ArrowLeft" && otpInputs[index - 1]) {
        otpInputs[index - 1].focus();
      }
      if (event.key === "ArrowRight" && otpInputs[index + 1]) {
        otpInputs[index + 1].focus();
      }
    });

    input.addEventListener("focus", function () {
      input.select();
    });
  });

  document
    .getElementById("otp-form")
    .addEventListener("submit", function (event) {
      event.preventDefault();
      showOtpError("");
      var code = otpInputs
        .map(function (input) {
          return input.value;
        })
        .join("");

      if (countdownValue <= 0) {
        showOtpError("This code expired. Request a fresh demo code to continue.");
        return;
      }
      if (code.length !== 6) {
        showOtpError("Enter all six digits to verify.");
        return;
      }
      if (code !== DEMO_OTP) {
        showOtpError("That code does not match the demo code.");
        return;
      }

      setView("sign-in");
      showNotice("Identity verified. You can now sign in with your new password.");
    });

  document.getElementById("resend-button").addEventListener("click", function () {
    if (countdownValue > 0) return;
    otpInputs.forEach(function (input) {
      input.value = "";
    });
    showOtpError("");
    startCountdown();
    window.setTimeout(function () {
      if (otpInputs[0]) otpInputs[0].focus();
    }, 100);
  });

  document.getElementById("google-button").addEventListener("click", function () {
    resetNotice();
    showNotice("Google sign-in is ready for the live workspace connection.");
  });

  document
    .getElementById("preview-toggle")
    .addEventListener("click", function () {
      var preview = document.getElementById("workspace-preview");
      var grid = document.getElementById("content-grid");
      var label = document.getElementById("preview-toggle-label");
      var isHidden = preview.hidden;
      preview.hidden = !isHidden;
      grid.classList.toggle("preview-hidden", !isHidden);
      label.textContent = isHidden
        ? "Back to dashboard"
        : "Show workspace preview";
    });
})();