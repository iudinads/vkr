document.addEventListener("DOMContentLoaded", function() {
    // Элементы интерфейса
    const mainButtons = document.querySelector(".action-buttons");
    const btnOpenFillOptions = document.getElementById("btn-open-fill-options");
    const fillOptionsContainer = document.getElementById("fill-options-container");
    const btnBack = document.getElementById("btn-back");
    const btnFill = document.getElementById("btn-fill");
    const resultMessage = document.getElementById("result-message");

    // Показ опций заполнения
    btnOpenFillOptions.addEventListener("click", function() {
        mainButtons.style.display = "none";
        fillOptionsContainer.style.display = "block";
    });

    // Возврат к основным кнопкам
    btnBack.addEventListener("click", function() {
        fillOptionsContainer.style.display = "none";
        mainButtons.style.display = "flex";
        resetRadioButtons();
        updateFillButtonState();
    });

    // Сброс radio-кнопок
    function resetRadioButtons() {
        document.querySelectorAll("input[name='categorical'], input[name='quantitative']").forEach(function(input) {
            input.checked = false;
        });
    }

    // Обновление состояния кнопки заполнения
    function updateFillButtonState() {
        const catSelected = document.querySelector("input[name='categorical']:checked");
        const quantSelected = document.querySelector("input[name='quantitative']:checked");
        
        if (catSelected && quantSelected) {
            btnFill.disabled = false;
            btnFill.classList.remove("disabled");
        } else {
            btnFill.disabled = true;
            btnFill.classList.add("disabled");
        }
    }

    // Назначение обработчиков для radio-кнопок
    document.querySelectorAll("input[name='categorical'], input[name='quantitative']").forEach(function(input) {
        input.addEventListener("change", updateFillButtonState);
    });

    // Обработка заполнения пропусков
    btnFill.addEventListener("click", function() {
        if (btnFill.disabled) return;

        const catValue = document.querySelector("input[name='categorical']:checked").value;
        const quantValue = document.querySelector("input[name='quantitative']:checked").value;

        const requestData = {
            categorical: catValue,
            quantitative: quantValue,
            n_neighbors: 5
        };

        // Показ спиннера
        document.getElementById("spinner-missing").style.display = "flex";

        fetch("/fill_missing", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            showResultMessage(data.message, data.status);
            resetInterface();
        })
        .catch(err => {
            showResultMessage("Ошибка при выполнении операции", "error");
            resetInterface();
        });
    });

    // Показать сообщение о результате
    function showResultMessage(message, status) {
        const resultElement = document.getElementById("result-message-missing");
        resultElement.textContent = message;
        resultElement.className = `result-message ${status}`;
        resultElement.style.display = "block";
    }

    // Сброс интерфейса
    function resetInterface() {
        document.getElementById("spinner-missing").style.display = "none";
        fillOptionsContainer.style.display = "none";
        mainButtons.style.display = "flex";
        resetRadioButtons();
        updateFillButtonState();
    }

    // Удаление пропусков
    document.querySelector(".btn-remove-missing").addEventListener("click", function(e) {
        e.preventDefault();
        const spinner = document.getElementById("spinner-missing");
        const resultElement = document.getElementById("result-message-missing");
        
        resultElement.style.display = "none";
        spinner.style.display = "flex";

        $.ajax({
            url: "/remove-missing", 
            type: "POST",
            dataType: "json",
            success: function(response) {
                spinner.style.display = "none";
                showResultMessage(response.message, 
                    response.message.toLowerCase().includes("ошибка") ? "error" : "success");
            },
            error: function() {
                spinner.style.display = "none";
                showResultMessage("Ошибка при выполнении запроса", "error");
            }
        });
    });

    // Удаление дубликатов
    document.querySelector(".btn-remove-duplicates").addEventListener("click", function(e) {
        e.preventDefault();
        const spinner = document.getElementById("spinner-duplicates");
        const resultElement = document.getElementById("result-message-duplicates");
        
        resultElement.style.display = "none";
        spinner.style.display = "flex";

        $.ajax({
            url: "/remove-duplicates", 
            type: "POST",
            dataType: "json",
            success: function(response) {
                spinner.style.display = "none";
                resultElement.textContent = response.message;
                resultElement.className = `result-message ${
                    response.message.toLowerCase().includes("ошибка") ? "error" : "success"
                }`;
                resultElement.style.display = "block";
            },
            error: function() {
                spinner.style.display = "none";
                resultElement.textContent = "Ошибка при выполнении запроса";
                resultElement.className = "result-message error";
                resultElement.style.display = "block";
            }
        });
    });

    // Инициализация
    updateFillButtonState();
});