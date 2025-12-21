VERSION=$(python -c "from AppConfig import AppConfig; print(AppConfig.App.VERSION)")
NAME=$(python -c "from AppConfig import AppConfig; print(AppConfig.App.NAME)")
ICON_PATH="$(pwd)/icon.ico"
APP_FILE="App.py"

# windows
pyinstaller \
--onefile \
--name "${NAME}_${VERSION}" \
--noconsole \
--distpath ./dist \
--workpath ./build \
--icon icon.ico \
--add-data "icon.ico;." \
${APP_FILE}

# macos
pyinstaller \
--onedir \
--name "${NAME}_${VERSION}" \
--noconsole \
--distpath ./dist \
--workpath ./build \
--icon icon.icns \
--add-data "icon.icns:." \
${APP_FILE}