set -e

echo "🚀 Setting up GitLab AI Code Reviewer"

# Check for required environment variables
required_vars=("GITLAB_URL" "GITLAB_TOKEN" "OPENAI_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=($var)
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Missing required environment variables: ${missing_vars[*]}"
    echo "Please set them in your .env file"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p storage/vectordb logs config

# Copy example configurations if not exists
if [ ! -f config/config.yaml ]; then
    echo "📋 Creating config.yaml from example..."
    cp config/config.yaml.example config/config.yaml
fi

if [ ! -f config/projects.yaml ]; then
    if [ -f config/projects.yaml.example ]; then
        echo "📋 Creating projects.yaml from example..."
        cp config/projects.yaml.example config/projects.yaml
        echo "⚠️  Please edit config/projects.yaml with your actual project configurations"
    else
        echo "❌ projects.yaml.example not found. Please create config/projects.yaml manually"
        exit 1
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Initialize vector database
echo "🗄️ Initializing vector database..."
python -c "
from src.vector_store import ChromaDBStore
import os
store = ChromaDBStore(
    path='./storage/vectordb',
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
print('Vector database initialized successfully')
"

# Test GitLab connection
echo "🔗 Testing GitLab connection..."
python -c "
from src.gitlab_client import GitLabClient
import os
client = GitLabClient(
    os.getenv('GITLAB_URL'),
    os.getenv('GITLAB_TOKEN')
)
print(f'Successfully connected as: {client.current_user.username}')
"

echo "✅ Setup complete! You can now:"
echo "  1. Edit config/projects.yaml to add your projects"
echo "  2. Run the service: python -m src.main"
echo "  3. Or use Docker: docker-compose up -d"