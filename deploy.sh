#!/usr/bin/env bash
set -euo pipefail

# VPS_ЗАМЕНИТЕ_МЕНЯ: публичный IP вашего VPS
VPS_IP="93.183.81.159"
# VPS_ЗАМЕНИТЕ_МЕНЯ: ваш домен
DOMAIN="mlpreprocessing.ru"
# VPS_ЗАМЕНИТЕ_МЕНЯ: SSH-пользователь на сервере
SSH_USER="root"
# VPS_ЗАМЕНИТЕ_МЕНЯ: email для certbot (обязательно реальный)
CERTBOT_EMAIL="diyudina033@gmail.com"
# VPS_ЗАМЕНИТЕ_МЕНЯ: путь проекта на VPS
REMOTE_APP_DIR="/opt/vkr"
GIT_REPO="git@github.com:iudinads/vkr.git"
# VPS_ЗАМЕНИТЕ_МЕНЯ: HTTPS URL репозитория (используется как fallback, если SSH-ключи не настроены на VPS)
GIT_REPO_HTTPS="https://github.com/iudinads/vkr.git"

if [[ "$VPS_IP" == *"xxx"* ]]; then
  echo "Ошибка: заполните VPS_IP в deploy.sh"
  exit 1
fi

if [[ "$CERTBOT_EMAIL" == "admin@example.com" ]]; then
  echo "Ошибка: заполните CERTBOT_EMAIL в deploy.sh"
  exit 1
fi

echo "Подключение к ${SSH_USER}@${VPS_IP} и запуск деплоя..."

ssh "${SSH_USER}@${VPS_IP}" \
  "DOMAIN='${DOMAIN}' CERTBOT_EMAIL='${CERTBOT_EMAIL}' REMOTE_APP_DIR='${REMOTE_APP_DIR}' GIT_REPO='${GIT_REPO}' GIT_REPO_HTTPS='${GIT_REPO_HTTPS}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

echo "[1/9] Обновление системы..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

echo "[2/9] Установка необходимых пакетов..."
apt-get install -y ca-certificates curl gnupg lsb-release ufw git nginx certbot openssl

if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if ! docker compose version >/dev/null 2>&1; then
  apt-get install -y docker-compose-plugin
fi

systemctl enable --now docker

echo "[3/9] Клонирование/обновление репозитория..."
mkdir -p "${REMOTE_APP_DIR}"
if [ ! -d "${REMOTE_APP_DIR}/.git" ]; then
  if git clone "${GIT_REPO}" "${REMOTE_APP_DIR}"; then
    echo "Клонирование по SSH успешно"
  else
    echo "SSH-клонирование не удалось, пробую HTTPS..."
    git clone "${GIT_REPO_HTTPS}" "${REMOTE_APP_DIR}"
  fi
else
  if ! git -C "${REMOTE_APP_DIR}" remote get-url origin | grep -q "${GIT_REPO_HTTPS}"; then
    git -C "${REMOTE_APP_DIR}" remote set-url origin "${GIT_REPO_HTTPS}" || true
  fi
  git -C "${REMOTE_APP_DIR}" fetch --all
  git -C "${REMOTE_APP_DIR}" reset --hard origin/main
fi

cd "${REMOTE_APP_DIR}"

echo "[4/9] Подготовка .env..."
if [ ! -f .env ]; then
  SECRET_KEY="$(openssl rand -hex 32)"
  cat > .env <<EOF
# VPS_ЗАМЕНИТЕ_МЕНЯ: строка подключения к вашей БД на VPS
DATABASE_URL=postgresql://localhost:5432/db
# Секретный ключ генерируется автоматически при первом деплое
SECRET_KEY=${SECRET_KEY}
# VPS_ЗАМЕНИТЕ_МЕНЯ: для продакшена обычно False
DEBUG=False
# VPS_ЗАМЕНИТЕ_МЕНЯ: ключ внешнего API, если используется
API_KEY=replace-me
# VPS_ЗАМЕНИТЕ_МЕНЯ: укажите реальные переменные окружения для продакшена
EOF
else
  echo ".env уже существует, не перезаписываю"
fi

echo "[5/9] Обновление nginx-конфига домена..."
if [ -f "nginx/default.conf" ]; then
  sed -i "s/server_name mlpreprocessing.online;/server_name ${DOMAIN};/g" nginx/default.conf || true
  sed -i "s#/etc/letsencrypt/live/mlpreprocessing.online/#/etc/letsencrypt/live/${DOMAIN}/#g" nginx/default.conf || true
fi

echo "[6/9] Настройка firewall..."
ufw allow 22/tcp || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true
ufw --force enable || true

echo "[7/9] Выпуск/обновление SSL сертификата..."
docker compose down || true
systemctl stop nginx || true
certbot certonly --standalone \
  --non-interactive \
  --agree-tos \
  --email "${CERTBOT_EMAIL}" \
  -d "${DOMAIN}" \
  --keep-until-expiring
systemctl start nginx || true

echo "[8/9] Запуск контейнеров..."
docker compose up -d --build

echo "[9/9] Проверка доступности API..."
sleep 5
if curl -kfsS "https://${DOMAIN}/" >/dev/null; then
  echo "Деплой завершен успешно"
  echo "Проверьте приложение: https://${DOMAIN}"
else
  echo "Предупреждение: HTTPS пока не отвечает, проверьте логи:"
  echo "docker compose logs --tail=100"
  exit 1
fi
REMOTE_SCRIPT

echo "Готово. Проверка: https://${DOMAIN}"
