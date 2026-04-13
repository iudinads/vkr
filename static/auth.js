document.addEventListener('DOMContentLoaded', function() {
  // Элементы основного модального окна авторизации
  const authModal = document.getElementById('authModal');
  const loginBtnNav = document.getElementById('loginBtnNav');
  const closeModal = document.querySelector('.close-modal');
  const statusElement = document.getElementById('authStatus');
  
  // Элементы форм
  const forgotPasswordLink = document.getElementById('forgotPassword');
  const sendRecoveryBtn = document.getElementById('sendRecoveryBtn');
  const recoveryStatus = document.getElementById('recoveryStatus');
  const backToLogin = document.getElementById('backToLogin');

  // Показать модальное окно авторизации
  if (loginBtnNav) {
    loginBtnNav.addEventListener('click', () => {
      authModal.style.display = 'flex';
      showForm('loginForm');
      resetForms();
    });
  }
  
  // Закрыть модальное окно авторизации
  closeModal.addEventListener('click', () => {
    authModal.style.display = 'none';
  });

  // Переключение между вкладками (Вход/Регистрация)
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      const tabName = this.dataset.tab;
      showForm(`${tabName}Form`);
      // Обновляем активность кнопок
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
    });
  });
  
  // Обработчики кнопок
  document.getElementById('loginBtn').addEventListener('click', handleLogin);
  document.getElementById('registerBtn').addEventListener('click', handleRegister);
  sendRecoveryBtn.addEventListener('click', handlePasswordRecovery);
  backToLogin.addEventListener('click', handleBackToLogin);

  // Открыть окно восстановления пароля
  forgotPasswordLink.addEventListener('click', (e) => {
    e.preventDefault();
    showForm('recoveryForm');
  });

  // Функции
  function showForm(formId) {
    // Скрываем все формы
    document.querySelectorAll('.auth-form').forEach(form => {
      form.style.display = 'none';
      form.classList.remove('active');
    });
    
    // Показываем выбранную форму
    const activeForm = document.getElementById(formId);
    activeForm.style.display = 'block';
    activeForm.classList.add('active');
    
    // Сбрасываем сообщения
    statusElement.innerHTML = ''; // NEW: используем innerHTML вместо textContent
    recoveryStatus.innerHTML = ''; // NEW: используем innerHTML вместо textContent
  }
  
  function resetForms() {
    // Сбрасываем значения полей
    document.querySelectorAll('.auth-form input').forEach(input => {
      input.value = '';
    });
    statusElement.innerHTML = ''; // NEW: используем innerHTML вместо textContent
  }
  
  // NEW: Полностью переработанная функция showStatus
  function showStatus(element, message, type = 'error') {
    // Полностью очищаем и пересоздаём элемент
    element.innerHTML = '';
    const statusDiv = document.createElement('div');
    statusDiv.className = `auth-status ${type}`;
    statusDiv.textContent = message;
    element.appendChild(statusDiv);
    
    // Автоматическое скрытие для успешных сообщений
    if (type === 'success') {
      setTimeout(() => {
        element.innerHTML = '';
      }, 3000);
    }
  }

  function handleBackToLogin(e) {
    e.preventDefault();
    showForm('loginForm');
    // Активируем соответствующую кнопку таба
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.tab === 'login') {
        btn.classList.add('active');
      }
    });
  }
  
  async function handleLogin() {
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    
    if (!email || !password) {
      showStatus(statusElement, 'Заполните все поля');
      return;
    }
    
    try {
      const response = await fetch('/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        showStatus(statusElement, 'Вход выполнен!', 'success');
        setTimeout(() => {
          authModal.style.display = 'none';
          window.location.reload();
        }, 1000);
      } else {
        showStatus(statusElement, data.message || 'Ошибка входа');
      }
    } catch (error) {
      showStatus(statusElement, 'Ошибка соединения');
      console.error('Ошибка:', error);
    }
  }
  
  async function handleRegister() {
    const name = document.getElementById('registerName').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('registerConfirmPassword').value;
    
    if (!name || !email || !password || !confirmPassword) {
      showStatus(statusElement, 'Заполните все поля');
      return;
    }
    
    if (password !== confirmPassword) {
      showStatus(statusElement, 'Пароли не совпадают');
      return;
    }
    
    try {
      const response = await fetch('/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name, email, password }),
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        showStatus(statusElement, 'Регистрация успешна! Вы вошли в систему.', 'success');
        setTimeout(() => {
          authModal.style.display = 'none';
          window.location.reload();
        }, 1000);
      } else {
        showStatus(statusElement, data.message || 'Ошибка регистрации');
      }
    } catch (error) {
      showStatus(statusElement, 'Ошибка соединения');
      console.error('Ошибка:', error);
    }
  }

  async function handlePasswordRecovery() {
    const email = document.getElementById('recoveryEmail').value;
    
    if (!email) {
      showStatus(recoveryStatus, 'Введите email');
      return;
    }

    try {
      const response = await fetch('/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        showStatus(recoveryStatus, data.message, 'success');
        setTimeout(() => {
          showForm('loginForm');
        }, 2000);
      } else {
        showStatus(recoveryStatus, data.message || 'Ошибка при восстановлении пароля');
        if (data.action === 'register') {
          showForm('registerForm');
          document.querySelector('[data-tab="register"]').classList.add('active');
          document.querySelector('[data-tab="login"]').classList.remove('active');
        }
      }
    } catch (error) {
      showStatus(recoveryStatus, 'Ошибка соединения');
      console.error('Ошибка:', error);
    }
  }
  
  // Закрытие при клике вне модального окна авторизации
  authModal.addEventListener('click', (e) => {
    if (e.target === authModal) {
      authModal.style.display = 'none';
    }
  });
});