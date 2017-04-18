% Primeiros passos com as competições de dados da Kaggle
% Créditos para Toshi Takeuchi
% https://blogs.mathworks.com/loren/2015/06/18/getting-started-with-kaggle-data-science-competitions/

% O objetivo da competição é prever os resultados de sobrevivência dos passageiros do Titanic. 
% Você usa os dados de treinamento do modelo e faz submissões as quais geram uma pontuação determinada pela precisão da previsão.

% Utilizei 2 arquivos .csv, um para treinamento e outro para testes.

% Framework: Matlab

% * |train.csv| treino
% * |test.csv| testes

%Importando dados em tabela através do Matlab

Train = readtable('train.csv','Format','%f%f%f%q%C%f%f%f%q%f%q%C');
Test = readtable('test.csv','Format','%f%f%q%C%f%f%f%q%f%q%C');
disp(Train(1:6,[2:3 3:9 10:12])) % define dados mostrado na visualização
%%

%    Legenda
%    1 - Survived 
%    0 - Didn't survive

%% Base de dados
%  https://www.kaggle.com/c/titanic/data

%% Testes

%Primeiro teste baseado na teoria de "mulheres e crianças primeiro"

disp(grpstats(Train(:,{'Survived','Sex'}), 'Sex')) % 74,2% das mulheres sobreviveram

% O Sexo é o principal fator de sobrevivência, mas não é suficiente.

gendermdl = grpstats(Train(:,{'Survived','Sex'}), {'Survived','Sex'})
all_female = (gendermdl.GroupCount('0_male') + gendermdl.GroupCount('1_female'))... 
    / sum(gendermdl.GroupCount) 

%Tirando uma média dos sobreviventes e a quantidade todal que sobreviveu a precisão foi de 78% levando em
%consideração que todas as mulheres sobreviveriam por causa da teoria. Ainda restam as crianças,
%então poderiamos estimar que os "homens" sobreviventes são crianças, mas ainda não é suficiente.

%Existe um problema de 177 passageiros sem idade na base de dados,
%removê-los não seria interessante pois perderiamos outros dados valiosos.
%Precisamos tratá-los posteriormente, assim como outros dados null ou
%zerados.

%% Tarifas
% O preço da tarifa apareceu fator que influenciou na sobrevivência dos passageiros como podemos ver abaixo.

figure
histogram(Train.Fare(Train.Survived == 0));        
hold on
histogram(Train.Fare(Train.Survived == 1),0:10:520) 
hold off
legend('Didn''t Survive', 'Survived')
title('The Titanic Passenger Fare Distribution')

% A maior parte das pessoas que pagaram valores de tarifas próximo de zero não sobreviveram
% O preço das Tarifas estão fortemente ligadas a classe social 

fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % get class average
disp(fare)

% Vamos preencher os valores vazios das cabines com uma média de acordo com
% a classe para tratar os dados faltantes, já que veremos a seguir que a classe social
% também determinou a sobrevivência em parte.

classesocial = grpstats(Train(:,{'Survived','Pclass'}), {'Survived','Pclass'}, {'Survived','Pclass'})

for i = 1:height(fare)
   
    Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
    Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
end



%% Cabine
% Alguns passageiros da terceira classe também possuem cabine, o que é um 
% privilégio dos passageiros de primeira classe. Então vamos tratar todas as
% pessoas que não possuem cabine como 0 e os que possui cabine como 1 e remover 
% as cabines da terceira classe. Podemos utilizar esses valores para preencher 
% dados faltantes relacionados no futuro.



% separa as strings por espaço em branco
train_cabins = cellfun(@strsplit, Train.Cabin, 'UniformOutput', false);
test_cabins = cellfun(@strsplit, Test.Cabin, 'UniformOutput', false);

% conta de acordo com a quantidade de espaços
Train.nCabins = cellfun(@length, train_cabins);
Test.nCabins = cellfun(@length, test_cabins);

% Apenas a primeira classe tem cabines
Train.nCabins(Train.Pclass ~= 1 & Train.nCabins > 1,:) = 1;
Test.nCabins(Test.Pclass ~= 1 & Test.nCabins > 1,:) = 1;

% Vazio é igual a 0 cabines
Train.nCabins(cellfun(@isempty, Train.Cabin)) = 0;
Test.nCabins(cellfun(@isempty, Test.Cabin)) = 0;

%% Local de embarque
% Para dois passageiros, não sabemos o seu porto de embarque. Usaremos 
% o valor mais frequente, "S" (Southampton), para 
% preencher os valores faltantes, o local também determinou a
% sobrevivência.

% pega valor mais frequente (moda)
freqVal = mode(Train.Embarked);

% Substitui valores faltantes
Train.Embarked(isundefined(Train.Embarked)) = freqVal;  
Test.Embarked(isundefined(Test.Embarked)) = freqVal;

%% Converte valores em double
Train.Embarked = double(Train.Embarked);
Test.Embarked = double(Test.Embarked);

Train.Sex = double(Train.Sex);
Test.Sex = double(Test.Sex);

%% Remove colunas que não terão relação na análise a partir daqui
Train(:,{'Name','Ticket','Cabin'}) = []; 
Test(:,{'Name','Ticket','Cabin'}) = [];

%% Tratando dados incomuns
% Segundo o gráfico da exibição, uma determinada margem idade teve mais 
% sobreviventes. Principalmente no grupo de menores de 5 anos (crianças de colo).
% Vamos agrupar os valores e considerar esse grupo como um atributo único. 

figure
histogram(Train.Age(Train.Survived == 0))   
hold on
histogram(Train.Age(Train.Survived == 1))  
hold off
legend('Didn''t Survive', 'Survived')
title('The Titanic Passenger Age Distribution')


Train.AgeGroup = double(discretize(Train.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));
Test.AgeGroup = double(discretize(Test.Age, [0:10:20 65 80], ...
    'categorical',{'child','teen','adult','senior'}));

% O Grupo Senior não teve sobreviventes, o que pode influenciar na predição
% dos dados de idades faltantes caso algum deles esteja neste grupo ou no
% primeiro, "child".


%% Random Forest and Boosted Trees
% Neste ponto, estamos prontos para aplicar alguns algoritmos de aprendizagem 
% de máquina no conjunto de dados. Um dos algoritmos mais populares no Kaggle é 
% um método de conjunto chamado Random Forest, e está disponível como  
% Bagged Trees no Matlab.

% Você pode utilizar outros aplicativos ou o Matlab para aproveitar os
% algoritmos, eu utilizei neste exemplo o Matlab.
